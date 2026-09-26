package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * YOLO mode (its own process, see the manifest): a deploy-pipeline YOLO model (yolo26n / yolo11n,
 * ../deploy) on the HTP, from the camera or the test images, with boxes, labels and an FPS panel.
 * Same camera path and extras as MainActivity, plus
 *   model   yolo26n (default), yolo11n, yolo26n-seg, yolo11n-seg (instance masks) or rfdetr_nano:
 *           <files>/models/<model>.onnx
 *   opts    YoloEngine options, e.g. "conf=0.25" (post=end2end for yolo26*, detr for rfdetr*,
 *           nms otherwise); "engine=tinygrad" runs <files>/models/<model>.tg, the tinygrad AOT OpenCL bundle
 *           (../tinygrad_aot), on the Adreno GPU instead of the HTP
 * The model buttons switch YOLO models in place (the engine re-inits its HTP session).
 */
public class YoloActivity extends MainActivity {
    static final String TAG = "YoloDemo";
    private volatile String wantModel;

    @Override
    int cameraMinWidth() {
        return YoloEngine.IN;
    }

    // ---- the engine, overridden by RtDetrActivity (same loop, another detector) ----
    String defaultModel() {
        return "yolo26n";
    }

    String engineInit(String model, String opts) {
        return YoloEngine.nativeInit(new File(getFilesDir(), "models").getAbsolutePath(),
                getApplicationInfo().nativeLibraryDir, model,
                model.startsWith("yolo26") || model.startsWith("rfdetr") || opts.contains("post=") ? opts
                        : "post=nms;" + opts);
    }

    Engine.Result newResult(String model) {
        if (model.contains("-seg")) {
            Engine.Result r = new Engine.Result(YoloEngine.SEG_MAX_DET, YoloEngine.MASK_SIDE, YoloEngine.T_N, 0.25f);
            r.colorByInstance = true;
            return r;
        }
        return new Engine.Result(YoloEngine.MAX_DET, false, YoloEngine.T_N, 0.25f);
    }

    int[] fitDims(int w, int h, int rot) {
        return YoloEngine.fitDims(w, h, rot);
    }

    void runYuv(Image img, int rot, Bitmap disp, Engine.Result r) {
        YoloEngine.runYuv(img, rot, disp, r);
    }

    void run(Bitmap rgba, Engine.Result r) {
        YoloEngine.run(rgba, r);
    }

    /** One line of per-stage times from the running averages of Result.times. */
    String stages(double[] avg) {
        // engine=tinygrad (tinygrad AOT OpenCL bundle on the Adreno) reports its inference in the same slot
        String opts = getIntent().getStringExtra("opts");
        String eng = opts != null && opts.contains("engine=tinygrad") ? "tinygrad gpu" : "htp";
        return String.format(Locale.US, "pre %.1f  %s %.2f  post %.2f ms", avg[YoloEngine.T_PRE], eng, avg[YoloEngine.T_HTP],
                avg[YoloEngine.T_POST]);
    }

    @Override
    boolean fastCamera() {
        return true;
    }

    @Override
    String activityKey() {
        return "yolo";
    }

    @Override
    void switchInPlace(String model) {
        wantModel = model;  // picked up by the loop
    }

    @Override
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        String model = getIntent().getStringExtra("model") != null ? getIntent().getStringExtra("model") : defaultModel();
        String yopts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : "";
        wantModel = model;
        if (cameraMode) startCamera();
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
        String cur = null;
        double initMs = 0;
        long[] done = new long[32];
        int nDone = 0;
        double[] avg = null;
        long seq = 0;
        while (running) {
            String want = wantModel;
            if (!want.equals(cur)) {  // (re)load: first frame or a model button
                overlay.setStats("loading " + want + "...");
                long t0 = System.nanoTime();
                String err = engineInit(want, yopts);
                initMs = (System.nanoTime() - t0) / 1e6;
                if (err != null) {
                    Log.e(TAG, "init failed: " + err);
                    overlay.setStats("INIT FAILED (" + want + "):\n" + err);
                    return;
                }
                Log.i(TAG, String.format(Locale.US, "init %s ok in %.0f ms", want, initMs));
                cur = want;
                nDone = 0;
            }
            Engine.Result r = newResult(cur);
            if (avg == null) avg = new double[r.times.length];
            try {
                if (cameraMode) {
                    Image img = reader != null ? reader.acquireLatestImage() : null;
                    if (img == null) {
                        try { Thread.sleep(2); } catch (InterruptedException e) { return; }
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = fitDims(img.getWidth(), img.getHeight(), rot);
                    r.frame = Bitmap.createBitmap(d[0], d[1], Bitmap.Config.ARGB_8888);
                    try {
                        runYuv(img, rot, r.frame, r);
                    } finally {
                        img.close();
                    }
                } else {
                    r.frame = decodeFit(images.get((int) (seq % images.size())), cameraMinWidth(), cameraMinWidth());
                    run(r.frame, r);
                }
                r.id = seq++;
            } catch (RuntimeException e) {
                Log.e(TAG, "run failed: " + e.getMessage());
                overlay.setStats("RUN FAILED:\n" + e.getMessage());
                return;
            }
            if (nDone == 0 || nDone == 9) {
                long up = SystemClock.uptimeMillis() - android.os.Process.getStartUptimeMillis();
                Log.i(TAG, String.format(Locale.US, "startup: result %d shown %d ms after process start (init %.0f ms)",
                        nDone + 1, up, initMs));
            }
            long now = System.nanoTime();
            done[nDone++ % done.length] = now;
            int k = Math.min(nDone, done.length);
            double fps = k > 1 ? (k - 1) / ((now - done[(nDone - k) % done.length]) / 1e9) : 0;
            for (int i = 0; i < avg.length; i++) avg[i] = nDone == 1 ? r.times[i] : 0.9 * avg[i] + 0.1 * r.times[i];
            int shown = 0;
            for (int i = 0; i < r.n; i++) if (r.scores[i] >= r.thresh) shown++;
            String s = String.format(Locale.US,
                    "%s  %s\nFPS %.1f (end to end)  inference %.1f ms -> %.0f FPS possible\n"
                    + "%s  detections %d%s",
                    cur, cameraMode ? "camera" : "images", fps, avg[0], 1000.0 / Math.max(avg[0], 1e-3), stages(avg), shown,
                    cameraMode ? "  rot " + frameRotation() : "");
            overlay.update(r, s);
            if (nDone % 30 == 0)
                Log.i(TAG, String.format(Locale.US, "%s frame %d fps %.1f total %.2f %s n %d", cur, r.id, fps, avg[0],
                        stages(avg), shown));
        }
    }
}
