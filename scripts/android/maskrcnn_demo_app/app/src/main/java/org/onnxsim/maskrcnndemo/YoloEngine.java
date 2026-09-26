package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/yolo_engine.cpp (a deploy-pipeline YOLO model on the HTP, post on the CPU). */
final class YoloEngine {
    static final int MAX_DET = 300, IN = 640;
    // -seg models: at most SEG_MAX_DET detections, each with a MASK_SIDE x MASK_SIDE mask over its box
    static final int SEG_MAX_DET = 100, MASK_SIDE = 40;
    static final int T_TOTAL = 0, T_PRE = 1, T_HTP = 2, T_POST = 3, T_N = 4;

    static {
        System.loadLibrary("yolo_demo");
    }

    /**
     * model: file stem in modelDir (yolo26n -> yolo26n.onnx; its EP-context model yolo26n.ctx0.onnx
     * is compiled on the first launch). A -seg model (a second, (1, 32, 160, 160) prototype output)
     * also returns instance masks. opts: "post=end2end|nms;conf=0.25;htp_performance_mode=burst".
     * Returns null on success, else the error. May be called again to switch models.
     */
    static native String nativeInit(String modelDir, String nativeLibDir, String model, String opts);
    static native int nativeRun(Bitmap rgba, float[] boxes, int[] labels, float[] scores, float[] masks, int maskSide,
                                float[] times);
    static native int nativeRunYuv(java.nio.ByteBuffer y, java.nio.ByteBuffer u, java.nio.ByteBuffer v, int yRowStride,
                                   int uvRowStride, int uvPixelStride, int w, int h, int rot, Bitmap disp,
                                   float[] boxes, int[] labels, float[] scores, float[] masks, int maskSide,
                                   float[] times);
    static native void nativeFitDims(int w, int h, int rot, int[] out);
    static native String nativeLastError();

    static int[] fitDims(int w, int h, int rot) {
        int[] d = new int[2];
        nativeFitDims(w, h, rot, d);
        return d;
    }

    /** Fills r (boxes in the display frame's pixels, 1-based COCO labels); throws on error. */
    static void runYuv(android.media.Image img, int rot, Bitmap disp, Engine.Result r) {
        android.media.Image.Plane[] p = img.getPlanes();
        int n = nativeRunYuv(p[0].getBuffer(), p[1].getBuffer(), p[2].getBuffer(), p[0].getRowStride(),
                p[1].getRowStride(), p[1].getPixelStride(), img.getWidth(), img.getHeight(), rot, disp, r.boxes,
                r.labels, r.scores, r.masks, r.maskSide, r.times);
        if (n < 0) throw new RuntimeException(nativeLastError());
        r.n = n;
    }

    static void run(Bitmap rgba, Engine.Result r) {
        int n = nativeRun(rgba, r.boxes, r.labels, r.scores, r.masks, r.maskSide, r.times);
        if (n < 0) throw new RuntimeException(nativeLastError());
        r.n = n;
    }
}
