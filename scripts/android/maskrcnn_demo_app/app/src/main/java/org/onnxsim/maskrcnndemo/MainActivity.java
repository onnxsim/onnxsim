package org.onnxsim.maskrcnndemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ExifInterface;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.WindowManager;
import android.widget.FrameLayout;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

/**
 * Live Mask R-CNN on the phone. Intent extras:
 *   mode    "camera" (default; back camera) or "images" (loops over the JPEGs in <files>/imgs)
 *   pipe    pipeline file in <files>/models (default pipe_e_opt.txt; pipe_e_opt_ctx.txt loads the
 *           HTP sessions from EP-context models)
 *   opts    engine options, e.g. "quant=lut;merge=seg2,seg4;pipeline=seg1" (maskrcnn_engine.cpp);
 *           "" = the plain e2e pipeline
 *   overlap images mode: decode/scale the next JPEG on a capture thread while the current frame
 *           runs (camera frames are always converted natively, from a latest-frame ImageReader)
 *
 * Orientation: the activity follows the device (fullUser: all four rotations, honoring the user's rotation lock). Camera frames arrive in the sensor's
 * orientation; each frame is rotated by (sensorOrientation - displayRotation) so the model always
 * sees it gravity-up, letterboxed into its landscape 1088x800 input, in the same native pass that
 * converts YUV and quantizes. Boxes/masks come back in that upright frame's coordinates, and the
 * displayed image is that same upright frame, so the overlay needs no further mapping.
 */
public class MainActivity extends Activity {
    private static final String TAG = "MaskRcnnDemo";
    OverlayView overlay;
    private TextureView preview;
    private HandlerThread camThread;
    private Handler camHandler;
    private CameraDevice camera;
    ImageReader reader;
    private int sensorOrientation = 90;
    private Size camSize;
    volatile boolean running = true;
    boolean cameraMode;
    private Thread worker;

    /** The demo's models: button label, then the activity (process) that runs it and its model extra. */
    static final String[][] MODELS = {{"Mask R-CNN", "", ""}, {"YOLO26n", "yolo", "yolo26n"}, {"YOLO11n", "yolo", "yolo11n"},
            {"YOLO26n-seg", "yolo", "yolo26n-seg"}, {"YOLO11n-seg", "yolo", "yolo11n-seg"},
            {"RT-DETR", "rtdetr", ""}, {"RF-DETR", "yolo", "rfdetr_nano"}, {"SAM", "sam", ""}, {"MCC 3D", "mcc", ""},
            {"Super-res", "sr", ""}, {"Game upscaling", "game", ""}};

    // The fastest measured configuration (README "Optimizations"); pass --es pipe pipe_e_opt.txt
    // --es opts "" for the original #1841 path.
    static final String DEFAULT_PIPE = "pipe_e_u8ra_ctx.txt";
    static final String DEFAULT_OPTS = "quant=lut;merge=seg2,seg4;pipeline=box_head";

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        FrameLayout root = new FrameLayout(this);
        overlay = new OverlayView(this);
        root.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        preview = new TextureView(this);
        root.addView(preview, new FrameLayout.LayoutParams(320, 240, Gravity.BOTTOM | Gravity.END));
        setContentView(root);

        String mode = getIntent().getStringExtra("mode");
        cameraMode = mode == null || mode.equals("camera");
        root.addView(modelBar(), new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.END));
        final String pipe = getIntent().getStringExtra("pipe") != null ? getIntent().getStringExtra("pipe") : DEFAULT_PIPE;
        final String opts = getIntent().getStringExtra("opts") != null ? getIntent().getStringExtra("opts") : DEFAULT_OPTS;
        final boolean overlap = getIntent().getBooleanExtra("overlap", false);
        if (!cameraMode) preview.setVisibility(android.view.View.GONE);
        if (cameraMode && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 1);
        worker = new Thread(() -> work(cameraMode, pipe, opts, overlap), "worker");
        worker.start();
    }

    /** The inference loop; YoloActivity runs its own. */
    void work(boolean cameraMode, String pipe, String opts, boolean overlap) {
        runLoop(cameraMode, pipe, opts, overlap);
    }

    /** Smallest 4:3 camera width that still covers the model input's long side. */
    int cameraMinWidth() {
        return Engine.IN_W;
    }

    /**
     * Ask the camera for its fastest fixed frame-rate range (auto-exposure otherwise drops to ~15
     * FPS indoors). Off for Mask R-CNN, which runs slower than the camera anyway.
     */
    boolean fastCamera() {
        return false;
    }

    /** Model selector: one button per entry of MODELS. */
    private android.view.View modelBar() {
        android.widget.LinearLayout bar = new android.widget.LinearLayout(this);
        for (String[] m : MODELS) {
            android.widget.Button b = new android.widget.Button(this);
            b.setText(m[0]);
            b.setAllCaps(false);
            b.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 13);
            b.setMinWidth(0);
            b.setMinimumWidth(0);
            b.setAlpha(0.8f);
            b.setOnClickListener(v -> switchTo(m[1], m[2]));
            bar.addView(b);
        }
        // camera <-> test images for the current model (switching models keeps the mode, so this is the only way
        // between them without adb)
        android.widget.Button mode = new android.widget.Button(this);
        mode.setText(cameraMode ? "Images" : "Camera");
        mode.setAllCaps(false);
        mode.setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 13);
        mode.setMinWidth(0);
        mode.setMinimumWidth(0);
        mode.setAlpha(0.8f);
        mode.setOnClickListener(v -> {  // the same activity again, in a new process (RelaunchActivity)
            android.content.Intent i = new android.content.Intent(this, RelaunchActivity.class);
            i.putExtra("target", getClass().getName());
            i.putExtra("pid", android.os.Process.myPid());
            i.putExtra("mode", cameraMode ? "images" : "camera");
            String model = getIntent().getStringExtra("model");
            if (model != null) i.putExtra("model", model);
            startActivity(i);
            finish();
        });
        bar.addView(mode, 0);  // first: visible without scrolling
        android.widget.HorizontalScrollView sv = new android.widget.HorizontalScrollView(this);
        sv.addView(bar);
        return sv;  // scrolls if the buttons don't fit (portrait)
    }

    /** This activity's key in MODELS ("" = Mask R-CNN). */
    String activityKey() {
        return "";
    }

    /** Switch to another model of this same activity (YoloActivity: another YOLO). */
    void switchInPlace(String model) {}

    /**
     * Switch model. Each engine runs in its own process (the native engines keep process-wide
     * HTP/DSP state, and onDestroy ends the process), so one engine is loaded at a time: switching
     * activities starts the other one and finishes (and kills) this one.
     */
    void switchTo(String key, String model) {
        if (key.equals(activityKey())) {
            switchInPlace(model);
            return;
        }
        Class<?> c = key.equals("yolo") ? YoloActivity.class
                : key.equals("rtdetr") ? RtDetrActivity.class
                : key.equals("sam") ? SamActivity.class
                : key.equals("mcc") ? MccActivity.class
                : key.equals("sr") ? SrActivity.class
                : key.equals("game") ? GameActivity.class : MainActivity.class;
        android.content.Intent i = new android.content.Intent(this, c);
        i.putExtra("mode", cameraMode ? "camera" : "images");
        if (!model.isEmpty()) i.putExtra("model", model);
        startActivity(i);
        finish();
    }

    private int displayRotationDegrees() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_90: return 90;
            case Surface.ROTATION_180: return 180;
            case Surface.ROTATION_270: return 270;
            default: return 0;
        }
    }

    /** Clockwise rotation that turns a back-camera sensor frame gravity-up for the current display rotation. */
    int frameRotation() {
        return (sensorOrientation - displayRotationDegrees() + 360) % 360;
    }

    // ---- images mode ------------------------------------------------------------------------
    /** Decode a JPEG upright (EXIF orientation applied) and scale it to fit 1088x800 in one pass. */
    static Bitmap decodeFit(File f) {
        return decodeFit(f, Engine.IN_W, Engine.IN_H);
    }

    /** Decode a JPEG upright and scale it to fit maxW x maxH. */
    static Bitmap decodeFit(File f, int maxW, int maxH) {
        Bitmap src = BitmapFactory.decodeFile(f.getPath());
        if (src == null) return null;
        int deg = 0;
        try {
            switch (new ExifInterface(f.getPath()).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                case ExifInterface.ORIENTATION_ROTATE_90: deg = 90; break;
                case ExifInterface.ORIENTATION_ROTATE_180: deg = 180; break;
                case ExifInterface.ORIENTATION_ROTATE_270: deg = 270; break;
                default: break;
            }
        } catch (java.io.IOException e) {
            Log.w(TAG, "exif " + f + ": " + e);
        }
        int uw = deg % 180 == 0 ? src.getWidth() : src.getHeight();
        int uh = deg % 180 == 0 ? src.getHeight() : src.getWidth();
        float ratio = Math.min((float) maxW / uw, (float) maxH / uh);
        Matrix m = new Matrix();
        m.postRotate(deg);
        m.postScale(ratio, ratio);
        Bitmap out = Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
        return out.getConfig() == Bitmap.Config.ARGB_8888 ? out : out.copy(Bitmap.Config.ARGB_8888, false);
    }

    private final Object frameLock = new Object();
    private Bitmap pending;
    private long pendingId = -1;

    private void captureLoop(List<File> images) {
        long id = 0;
        while (running) {
            Bitmap in = decodeFit(images.get((int) (id % images.size())));
            synchronized (frameLock) {
                while (pending != null && running) {  // process every image, in order
                    try { frameLock.wait(); } catch (InterruptedException e) { return; }
                }
                pending = in;
                pendingId = id++;
                frameLock.notifyAll();
            }
        }
    }

    // ---- main loop --------------------------------------------------------------------------
    private void runLoop(boolean cameraMode, String pipe, String opts, boolean overlapCapture) {
        File models = new File(getFilesDir(), "models");
        overlay.setStats("loading models (" + pipe + ")...");
        // the camera opens (a few hundred ms, on its own thread) while the models load
        if (cameraMode) startCamera();
        long t0 = System.nanoTime();
        String err = Engine.nativeInit(models.getAbsolutePath(), getApplicationInfo().nativeLibraryDir,
                new File(models, pipe).getAbsolutePath(), opts);
        double initMs = (System.nanoTime() - t0) / 1e6;
        if (err != null) {
            Log.e(TAG, "init failed: " + err);
            overlay.setStats("INIT FAILED:\n" + err);
            return;
        }
        Log.i(TAG, String.format(Locale.US, "init ok in %.0f ms (%s, opts '%s', overlap %b)", initMs, pipe, opts,
                overlapCapture));
        final List<File> images = new ArrayList<>();
        if (!cameraMode) {
            File[] fs = new File(getFilesDir(), "imgs").listFiles();
            if (fs != null) for (File f : fs) if (f.getName().endsWith(".jpg")) images.add(f);
            Collections.sort(images);
            if (images.isEmpty()) {
                overlay.setStats("no images in " + new File(getFilesDir(), "imgs"));
                return;
            }
            if (overlapCapture) new Thread(() -> captureLoop(images), "capture").start();
        }
        long[] done = new long[32];
        int nDone = 0;
        double[] stage = new double[Engine.T_N];
        HashMap<Long, Bitmap> inFlight = new HashMap<>();
        long seq = 0;
        String label = (cameraMode ? "camera" : "images" + (overlapCapture ? " +overlap" : ""))
                + (opts.isEmpty() ? "" : " [" + opts + "]");
        while (running) {
            Engine.Result r = new Engine.Result();
            long got;
            try {
                if (cameraMode) {
                    Image img = reader != null ? reader.acquireLatestImage() : null;
                    if (img == null) {
                        try { Thread.sleep(3); } catch (InterruptedException e) { return; }
                        continue;
                    }
                    int rot = frameRotation();
                    int[] d = Engine.fitDims(img.getWidth(), img.getHeight(), rot);
                    Bitmap disp = Bitmap.createBitmap(d[0], d[1], Bitmap.Config.ARGB_8888);
                    long id = seq++;
                    inFlight.put(id, disp);
                    try {
                        got = Engine.submitYuv(img, rot, disp, id, r);
                    } finally {
                        img.close();
                    }
                } else {
                    Bitmap in;
                    long id;
                    if (overlapCapture) {
                        synchronized (frameLock) {
                            while (pending == null && running) {
                                try { frameLock.wait(); } catch (InterruptedException e) { return; }
                            }
                            if (!running) return;
                            in = pending;
                            id = pendingId;
                            pending = null;
                            frameLock.notifyAll();
                        }
                    } else {
                        in = decodeFit(images.get((int) (seq % images.size())));
                        id = seq++;
                    }
                    inFlight.put(id, in);
                    got = Engine.submit(in, id, r);
                }
            } catch (RuntimeException e) {
                Log.e(TAG, "run failed: " + e.getMessage());
                overlay.setStats("RUN FAILED:\n" + e.getMessage());
                return;
            }
            if (got < 0) continue;  // pipeline filling
            if (nDone == 0 || nDone == 9) {  // startup: time to the first / tenth shown result
                long up = SystemClock.uptimeMillis() - android.os.Process.getStartUptimeMillis();
                Log.i(TAG, String.format(Locale.US, "startup: result %d shown %d ms after process start (init %.0f ms)",
                        nDone + 1, up, initMs));
            }
            final long g = got;
            r.frame = inFlight.remove(got);
            inFlight.keySet().removeIf(x -> x < g);
            long now = System.nanoTime();
            done[nDone++ % done.length] = now;
            int k = Math.min(nDone, done.length);
            double fps = k > 1 ? (k - 1) / ((now - done[(nDone - k) % done.length]) / 1e9) : 0;
            for (int i = 0; i < Engine.T_N; i++) stage[i] = nDone == 1 ? r.times[i] : 0.9 * stage[i] + 0.1 * r.times[i];
            int shown = 0;
            for (int i = 0; i < r.n; i++) if (r.scores[i] >= OverlayView.SCORE_THRESH) shown++;
            String s = String.format(Locale.US,
                    "%s\nFPS %.2f  latency %.1f ms (avg %.1f)  stage A %.1f  B %.1f  wait %.1f\n"
                    + "pre %.1f  backbone %.1f  rpn %.1f  roialign %.1f  heads %.1f  cpu %.1f ms\n"
                    + "detections %d  %s%s",
                    label, fps, r.times[Engine.T_TOTAL], stage[Engine.T_TOTAL], stage[Engine.T_STAGE_A],
                    stage[Engine.T_STAGE_B], stage[Engine.T_WAIT], stage[Engine.T_PRE], stage[Engine.T_BACKBONE],
                    stage[Engine.T_RPN], stage[Engine.T_ROI], stage[Engine.T_HEADS], stage[Engine.T_CPU], shown, pipe,
                    cameraMode ? "  rot " + frameRotation() : "");
            overlay.update(r, s);
            if (nDone % 10 == 0)
                Log.i(TAG, String.format(Locale.US, "frame %d fps %.2f lat %.1f %s", got, fps, r.times[Engine.T_TOTAL],
                        Arrays.toString(r.times)));
            if (!cameraMode) {  // per-image output checksum, to A/B pipeline options for equality
                double cs = 0;
                for (int i = 0; i < r.n; i++) {
                    cs += r.scores[i] + r.labels[i];
                    for (int j = 0; j < 4; j++) cs += r.boxes[4 * i + j];
                    for (int j = 0; j < 784; j += 97) cs += r.masks[784 * i + j];
                }
                Log.i(TAG, String.format(Locale.US, "check img %d n %d sum %.6f", got % images.size(), r.n, cs));
            }
        }
    }

    // ---- camera -----------------------------------------------------------------------------
    void startCamera() {
        camThread = new HandlerThread("cam");
        camThread.start();
        camHandler = new Handler(camThread.getLooper());
        new Handler(Looper.getMainLooper()).post(() -> {
            if (preview.isAvailable()) openCamera();
            else preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override public void onSurfaceTextureAvailable(SurfaceTexture s, int w, int h) { openCamera(); }
                @Override public void onSurfaceTextureSizeChanged(SurfaceTexture s, int w, int h) { configureTransform(); }
                @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture s) { return true; }
                @Override public void onSurfaceTextureUpdated(SurfaceTexture s) {}
            });
        });
    }

    /**
     * Keeps the small live preview upright: resize the thumbnail to the upright aspect and set the
     * TextureView transform for the display rotation (the classic Camera2 sample recipe, generalized
     * to the sensor orientation). Only the thumbnail uses this; inference frames come from the
     * ImageReader and are rotated natively.
     */
    private void configureTransform() {
        if (camSize == null) return;
        int rot = frameRotation();
        int longSide = 320, shortSide = 240;
        int vw = rot % 180 == 0 ? longSide : shortSide, vh = rot % 180 == 0 ? shortSide : longSide;
        FrameLayout.LayoutParams lp = (FrameLayout.LayoutParams) preview.getLayoutParams();
        if (lp.width != vw || lp.height != vh) {
            lp.width = vw;
            lp.height = vh;
            preview.setLayoutParams(lp);
        }
        // the buffer is drawn stretched to the view; undo the stretch and rotate about the centre
        Matrix m = new Matrix();
        RectF view = new RectF(0, 0, vw, vh);
        float cx = view.centerX(), cy = view.centerY();
        int disp = displayRotationDegrees();
        if (disp == 90 || disp == 270) {
            RectF buf = new RectF(0, 0, camSize.getHeight(), camSize.getWidth());
            buf.offset(cx - buf.centerX(), cy - buf.centerY());
            m.setRectToRect(view, buf, Matrix.ScaleToFit.FILL);
            float s = Math.max((float) vh / camSize.getHeight(), (float) vw / camSize.getWidth());
            m.postScale(s, s, cx, cy);
            // SurfaceTexture already presents the buffer in the device's natural (portrait) orientation;
            // landscape display rotations turn it back by 90 * (rotation index - 2)
            m.postRotate(90 * (disp / 90 - 2), cx, cy);
        } else if (disp == 180) {
            m.postRotate(180, cx, cy);
        }
        preview.setTransform(m);
    }

    @Override
    public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        configureTransform();
    }

    private void openCamera() {
        try {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                overlay.setStats("camera permission missing (adb shell pm grant org.onnxsim.maskrcnndemo android.permission.CAMERA)");
                return;
            }
            CameraManager cm = (CameraManager) getSystemService(CAMERA_SERVICE);
            String id = null;
            for (String c : cm.getCameraIdList()) {
                Integer f = cm.getCameraCharacteristics(c).get(CameraCharacteristics.LENS_FACING);
                if (f != null && f == CameraCharacteristics.LENS_FACING_BACK) { id = c; break; }
            }
            if (id == null) id = cm.getCameraIdList()[0];
            CameraCharacteristics ch = cm.getCameraCharacteristics(id);
            Integer so = ch.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation = so != null ? so : 90;
            StreamConfigurationMap map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            // smallest 4:3 YUV size that still covers the model input's long side
            Size best = null;
            for (Size s : map.getOutputSizes(ImageFormat.YUV_420_888)) {
                if (s.getWidth() * 3 != s.getHeight() * 4 || s.getWidth() < cameraMinWidth()) continue;
                if (best == null || s.getWidth() < best.getWidth()) best = s;
            }
            camSize = best != null ? best : new Size(1440, 1080);
            reader = ImageReader.newInstance(camSize.getWidth(), camSize.getHeight(), ImageFormat.YUV_420_888, 3);
            configureTransform();
            cm.openCamera(id, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice d) {
                    camera = d;
                    try {
                        SurfaceTexture st = preview.getSurfaceTexture();
                        st.setDefaultBufferSize(camSize.getWidth(), camSize.getHeight());
                        Surface surf = new Surface(st);
                        CaptureRequest.Builder rb = d.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        rb.addTarget(surf);
                        rb.addTarget(reader.getSurface());
                        if (fastCamera()) {
                            android.util.Range<Integer> best = null;
                            android.util.Range<Integer>[] rs = ch.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
                            if (rs != null)
                                for (android.util.Range<Integer> r : rs)
                                    if (best == null || r.getLower() > best.getLower()
                                            || (r.getLower().equals(best.getLower()) && r.getUpper() > best.getUpper()))
                                        best = r;
                            if (best != null) {
                                rb.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, best);
                                Log.i(TAG, "camera fps range " + best);
                            }
                        }
                        d.createCaptureSession(Arrays.asList(surf, reader.getSurface()), new CameraCaptureSession.StateCallback() {
                            @Override public void onConfigured(CameraCaptureSession s) {
                                try { s.setRepeatingRequest(rb.build(), null, camHandler); }
                                catch (Exception e) { Log.e(TAG, "repeat", e); }
                            }
                            @Override public void onConfigureFailed(CameraCaptureSession s) { Log.e(TAG, "configure failed"); }
                        }, camHandler);
                        Log.i(TAG, "camera " + camSize + " sensorOrientation " + sensorOrientation);
                    } catch (Exception e) { Log.e(TAG, "session", e); }
                }
                @Override public void onDisconnected(CameraDevice d) { d.close(); }
                @Override public void onError(CameraDevice d, int e) { Log.e(TAG, "camera error " + e); d.close(); }
            }, camHandler);
        } catch (Exception e) {
            Log.e(TAG, "openCamera", e);
        }
    }

    @Override
    protected void onDestroy() {
        running = false;
        synchronized (frameLock) { frameLock.notifyAll(); }
        if (camera != null) camera.close();
        if (camThread != null) camThread.quitSafely();
        super.onDestroy();
        // the native engine keeps DSP/HTP sessions; end the process so a relaunch starts clean
        android.os.Process.killProcess(android.os.Process.myPid());
    }
}
