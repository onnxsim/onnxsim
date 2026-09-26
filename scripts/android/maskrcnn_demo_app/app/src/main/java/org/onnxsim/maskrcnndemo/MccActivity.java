package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * MCC single-image 3D reconstruction (its own process, see the manifest; mcc_engine.cpp): tap an
 * object to segment it (SAM), then "3D" runs MoGe-2 (monocular point map) and MCC (encoder + a
 * coarse-to-fine decoder) and shows the reconstructed colored points, turned by dragging.
 *   camera: live preview; a tap captures the object under it: freezes that frame (SAM encoder), segments it and
 *           reconstructs it in 3D. Taps on the frozen photo re-segment ("3D" then rebuilds); "Live" goes back.
 *   images: the test images, center-cropped to 3:4 / 4:3; "Next image" moves on.
 * Extras: "tap" ("fx,fy", fractions of the frame) taps after each encode and "recon" (boolean) then
 * reconstructs (scripted runs); "image" (a file name in imgs) starts there; "opts" (mcc_engine.cpp: gran,
 * levels, lo, thr, dump=1).
 */
public class MccActivity extends MainActivity {
    private static final String TAG = "MccDemo";
    private final ConcurrentLinkedQueue<float[]> taps = new ConcurrentLinkedQueue<>();
    private volatile boolean frozen, next, reconRequested;
    private boolean captureRecon;  // worker thread only
    private PointCloudView cloud;
    private Button view3d;

    @Override
    String activityKey() {
        return "mcc";
    }

    @Override
    int cameraMinWidth() {
        return MccEngine.LONG;
    }

    @Override
    boolean fastCamera() {
        return true;
    }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        FrameLayout root = (FrameLayout) overlay.getParent();
        cloud = new PointCloudView(this);
        cloud.setVisibility(View.GONE);
        root.addView(cloud, 1, new FrameLayout.LayoutParams(-1, -1));  // over the photo, under the buttons
        Button live = new Button(this);
        live.setText(cameraMode ? "Live" : "Next image");
        live.setAllCaps(false);
        live.setAlpha(0.8f);
        live.setOnClickListener(v -> {
            showCloud(false);
            if (cameraMode) frozen = false;
            else next = true;
        });
        view3d = new Button(this);
        view3d.setText("3D");
        view3d.setAllCaps(false);
        view3d.setAlpha(0.8f);
        view3d.setOnClickListener(v -> {
            if (cloud.getVisibility() == View.VISIBLE) showCloud(false);
            else reconRequested = true;
        });
        LinearLayout bar = new LinearLayout(this);
        bar.addView(live);
        bar.addView(view3d);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        lp.topMargin = (int) (72 * getResources().getDisplayMetrics().density);  // below the model buttons
        root.addView(bar, lp);
        overlay.setOnTouchListener((v, e) -> {
            if (e.getAction() != MotionEvent.ACTION_UP) return true;
            float[] f = overlay.toFrame(e.getX(), e.getY());
            if (f != null) taps.add(f);
            return true;
        });
    }

    private void showCloud(boolean on) {
        runOnUiThread(() -> {
            cloud.setVisibility(on ? View.VISIBLE : View.GONE);
            view3d.setText(on ? "Photo" : "3D");
        });
    }

    /** The photo with the mask blended in, for the 3D view's corner. */
    private static Bitmap thumbnail(Bitmap frame, Bitmap mask) {
        Bitmap t = frame.copy(Bitmap.Config.ARGB_8888, true);
        if (mask != null) new Canvas(t).drawBitmap(mask, 0, 0, new Paint());
        return t;
    }

    @Override
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        String mopts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        String autoTap = getIntent().getStringExtra("tap");
        boolean autoRecon = getIntent().getBooleanExtra("recon", false);
        float gran = 0.1f;
        for (String kv : mopts.split(";"))
            if (kv.startsWith("gran=")) gran = Float.parseFloat(kv.substring(5));
        if (cameraMode) startCamera();
        overlay.setStats("loading SAM + MCC (the first launch compiles the HTP graphs, a few minutes)...");
        long t0 = System.nanoTime();
        String err = MccEngine.nativeInit(new File(getFilesDir(), "models").getAbsolutePath(),
                getApplicationInfo().nativeLibraryDir, mopts);
        double initMs = (System.nanoTime() - t0) / 1e6;
        if (err != null) {
            Log.e(TAG, "init failed: " + err);
            overlay.setStats("INIT FAILED:\n" + err);
            return;
        }
        Log.i(TAG, String.format(Locale.US, "init ok in %.0f ms", initMs));
        List<File> images = new ArrayList<>();
        if (!cameraMode) {
            File[] fs = new File(getFilesDir(), "imgs").listFiles();
            if (fs != null) for (File f : fs) if (f.getName().endsWith(".jpg")) images.add(f);
            Collections.sort(images);
            if (images.isEmpty()) {
                overlay.setStats("no images in " + new File(getFilesDir(), "imgs"));
                return;
            }
        }
        int img = 0;
        for (int i = 0; i < images.size(); i++)
            if (images.get(i).getName().equals(getIntent().getStringExtra("image"))) img = i;
        float[] et = new float[1], dt = new float[1], iou = new float[4], rt = new float[6];
        int[] rc = new int[3];
        byte[] mask = null;
        Engine.Result r = null;  // the working image on screen
        Bitmap maskBmp = null;
        float mx = -1, my = -1, encMs = 0, decMs = 0;
        int slot = 0;
        boolean needEncode = !cameraMode, autoReconDone = false;
        String photoLine = "";
        try {
            while (running) {
                boolean changed = false;
                if (!cameraMode && next) {
                    next = false;
                    img = (img + 1) % images.size();
                    needEncode = true;
                }
                float[] tap = taps.poll();
                if (cameraMode && !frozen && tap != null) {  // a tap on the live preview: freeze, encode, segment, 3D
                    frozen = true;
                    needEncode = true;
                    captureRecon = true;
                }
                if (cameraMode && (!frozen || needEncode)) {
                    Image im = reader != null ? reader.acquireLatestImage() : null;
                    if (im == null) {
                        if (tap != null) taps.add(tap);  // retry on the next frame
                        Thread.sleep(2);
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = rot % 180 != 0 ? MccEngine.workDims(im.getHeight(), im.getWidth())
                            : MccEngine.workDims(im.getWidth(), im.getHeight());
                    Engine.Result nr = new Engine.Result(1, false, 1, 1f);
                    nr.frame = Bitmap.createBitmap(d[0], d[1], Bitmap.Config.ARGB_8888);
                    try {
                        MccEngine.check(MccEngine.nativeYuv(im.getPlanes()[0].getBuffer(), im.getPlanes()[1].getBuffer(),
                                im.getPlanes()[2].getBuffer(), im.getPlanes()[0].getRowStride(),
                                im.getPlanes()[1].getRowStride(), im.getPlanes()[1].getPixelStride(), im.getWidth(),
                                im.getHeight(), rot, nr.frame, needEncode, et));
                    } finally {
                        im.close();
                    }
                    r = nr;
                    maskBmp = null;
                    mx = -1;
                    if (!needEncode) {
                        overlay.update(r, "MCC 3D  camera: live preview\ntap an object to capture it in 3D",
                                null, 0, 0, -1, -1);
                        continue;
                    }
                    needEncode = false;
                    changed = true;
                    encMs = et[0];
                    autoReconDone = false;
                } else if (needEncode) {
                    needEncode = false;
                    Engine.Result nr = new Engine.Result(1, false, 1, 1f);
                    nr.frame = MccEngine.working(decodeFit(images.get(img), 2048, 2048));
                    MccEngine.check(MccEngine.nativeEncode(nr.frame, et));
                    r = nr;
                    changed = true;
                    encMs = et[0];
                    maskBmp = null;
                    mx = -1;
                    autoReconDone = false;
                    Log.i(TAG, String.format(Locale.US, "encode %s: SAM encoder %.1f ms", images.get(img).getName(), et[0]));
                }
                if (tap == null && autoTap != null && mx < 0 && r != null) {  // scripted tap after each encode
                    String[] a = autoTap.split(",");
                    tap = new float[] {Float.parseFloat(a[0]) * r.frame.getWidth(), Float.parseFloat(a[1]) * r.frame.getHeight()};
                }
                if (tap != null && r != null && cloud.getVisibility() != View.VISIBLE) {
                    int w = r.frame.getWidth(), h = r.frame.getHeight();
                    if (mask == null || mask.length != w * h) mask = new byte[w * h];
                    slot = MccEngine.nativeSegment(tap[0], tap[1], mask, iou, dt);
                    if (slot < 0) throw new RuntimeException(MccEngine.nativeLastError());
                    decMs = dt[0];
                    int[] px = new int[w * h];
                    for (int i = 0; i < w * h; i++) px[i] = mask[i] != 0 ? 0x9000A0FF : 0;
                    maskBmp = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888);
                    mx = tap[0];
                    my = tap[1];
                    changed = true;
                    Log.i(TAG, String.format(Locale.US, "segment at (%.0f, %.0f): %.1f ms, slot %d, iou %.3f", mx, my, decMs,
                            slot, iou[slot]));
                }
                if (changed) {
                    photoLine = String.format(Locale.US, "MCC 3D  %s\nSAM encoder %.1f ms, decoder %.1f ms per tap\n%s",
                            cameraMode ? "camera, frozen frame" : "images " + (img + 1) + "/" + images.size(), encMs, decMs,
                            mx < 0 ? "tap an object to segment it" : String.format(Locale.US,
                                    "mask IoU %.3f -- \"3D\" reconstructs it", iou[slot]));
                    overlay.update(r, photoLine, maskBmp, maskBmp != null ? maskBmp.getWidth() : 0,
                            maskBmp != null ? maskBmp.getHeight() : 0, mx, my);
                }
                if (captureRecon && mx >= 0) {  // the capturing tap: its mask straight into 3D
                    captureRecon = false;
                    reconRequested = true;
                }
                if (autoRecon && !autoReconDone && mx >= 0) {
                    autoReconDone = true;
                    reconRequested = true;
                }
                if (reconRequested) {
                    reconRequested = false;
                    if (mx < 0) {
                        overlay.setStats(photoLine + "\n(tap an object first)");
                        continue;
                    }
                    overlay.setStats(photoLine + "\nreconstructing: MoGe-2 -> MCC encoder -> decoder chunks...");
                    int n = MccEngine.nativeReconstruct(rt, rc);
                    if (n < 0) {  // e.g. no valid depth under the mask: say so, keep going
                        String e = MccEngine.nativeLastError();
                        Log.e(TAG, "recon failed: " + e);
                        overlay.setStats(photoLine + "\nreconstruction failed: " + e);
                        continue;
                    }
                    float[] xyz = new float[3 * n];
                    int[] col = new int[n];
                    MccEngine.nativePoints(xyz, col);
                    String s = String.format(Locale.US,
                            "MCC 3D  %d points (granularity %.2f)\nMoGe-2 %.0f ms (in the background, waited %.0f ms)\n"
                                    + "prep %.0f ms  MCC encoder %.0f ms\nMCC decoder %.0f ms (%d queries, %d x 1024)\n"
                                    + "total %.2f s   drag to turn, pinch to zoom",
                            n, gran, rt[0], rt[1], rt[2], rt[3], rt[4], rc[0], rc[1], rt[5] / 1000);
                    Log.i(TAG, String.format(Locale.US,
                            "recon: %d points; MoGe %.1f (waited %.1f) prep %.1f encoder %.1f decoder %.1f (%d queries, %d chunks) total %.1f ms",
                            n, rt[0], rt[1], rt[2], rt[3], rt[4], rc[0], rc[1], rt[5]));
                    cloud.setPoints(xyz, col, gran, thumbnail(r.frame, maskBmp), s);
                    showCloud(true);
                    overlay.setStats(photoLine);
                }
                if (taps.isEmpty()) Thread.sleep(5);
            }
        } catch (InterruptedException e) {
            // shutting down
        } catch (RuntimeException e) {
            Log.e(TAG, "run failed: " + e.getMessage());
            overlay.setStats("RUN FAILED:\n" + e.getMessage());
        }
    }
}
