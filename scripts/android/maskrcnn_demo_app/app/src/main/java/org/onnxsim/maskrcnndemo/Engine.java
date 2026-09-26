package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/maskrcnn_engine.cpp (the PR #1841 e2e pipeline as a library). */
final class Engine {
    static final int MAX_DET = 100;
    static final int T_TOTAL = 0, T_PRE = 1, T_BACKBONE = 2, T_RPN = 3, T_ROI = 4, T_HEADS = 5, T_CPU = 6,
            T_STAGE_A = 7, T_STAGE_B = 8, T_WAIT = 9, T_N = 10;
    static final int IN_W = 1088, IN_H = 800;

    static {
        System.loadLibrary("maskrcnn_demo");
    }

    /**
     * Returns null on success, else an error message. opts: "key=value;..." (see
     * maskrcnn_engine.cpp: quant=lut, merge=seg2,seg4, pipeline=&lt;first stage-B step&gt;), "" for the
     * plain e2e pipeline.
     */
    static native String nativeInit(String modelDir, String nativeLibDir, String pipe, String opts);
    /** Submits frame id; returns the id of a finished frame (results written), -1 none yet, -2 error. */
    static native long nativeRun(Bitmap rgba, long id, float[] boxes, int[] labels, float[] scores, float[] masks,
                                 float[] times, int[] ndet);
    static native String nativeLastError();
    /**
     * Camera fast path: YUV_420_888 planes -> rotated (rot: clockwise degrees making the frame
     * upright), letterboxed, quantized backbone input, plus the upright frame into disp (RGBA, size
     * from fitDims). Same return convention as nativeRun.
     */
    static native long nativeRunYuv(java.nio.ByteBuffer y, java.nio.ByteBuffer u, java.nio.ByteBuffer v, int yRowStride,
                                    int uvRowStride, int uvPixelStride, int w, int h, int rot, Bitmap disp, long id,
                                    float[] boxes, int[] labels, float[] scores, float[] masks, float[] times, int[] ndet);
    static native void nativeFitDims(int w, int h, int rot, int[] out);

    static int[] fitDims(int w, int h, int rot) {
        int[] d = new int[2];
        nativeFitDims(w, h, rot, d);
        return d;
    }

    static long submitYuv(android.media.Image img, int rot, Bitmap disp, long id, Result r) {
        android.media.Image.Plane[] p = img.getPlanes();
        long got = nativeRunYuv(p[0].getBuffer(), p[1].getBuffer(), p[2].getBuffer(), p[0].getRowStride(),
                p[1].getRowStride(), p[1].getPixelStride(), img.getWidth(), img.getHeight(), rot, disp, id, r.boxes,
                r.labels, r.scores, r.masks, r.times, r.ndet);
        if (got == -2) throw new RuntimeException(nativeLastError());
        r.n = r.ndet[0];
        r.id = got;
        return got;
    }

    /**
     * One frame's output, in the displayed frame's pixel coordinates (for Mask R-CNN: model-input
     * pixels, image top-left aligned in 1088x800). Also used by the YOLO mode: no masks
     * (masks.length == 0) or, for the -seg models, maskSide x maskSide per detection; its own score
     * threshold and timing slots.
     */
    static final class Result {
        final float[] boxes, scores, masks, times;
        final int[] labels;
        final int[] ndet = new int[1];
        final float thresh;
        final int maskSide;  // each detection's mask: maskSide x maskSide probabilities over its box
        boolean colorByInstance;  // the overlay colours each detection apart (crowds of one class), not by class
        int n;
        long id;
        Bitmap frame;   // the (resized) frame the result belongs to

        Result() {
            this(MAX_DET, true, T_N, OverlayView.SCORE_THRESH);
        }

        Result(int maxDet, boolean withMasks, int nTimes, float thresh) {
            this(maxDet, withMasks ? 28 : 0, nTimes, thresh);
        }

        Result(int maxDet, int maskSide, int nTimes, float thresh) {
            this.maskSide = maskSide;
            boxes = new float[4 * maxDet];
            labels = new int[maxDet];
            scores = new float[maxDet];
            masks = new float[maskSide * maskSide * maxDet];
            times = new float[nTimes];
            this.thresh = thresh;
        }
    }

    /** Returns the finished frame's id (r filled), -1 if none yet; throws on error. */
    static long submit(Bitmap rgba, long id, Result r) {
        long got = nativeRun(rgba, id, r.boxes, r.labels, r.scores, r.masks, r.times, r.ndet);
        if (got == -2) throw new RuntimeException(nativeLastError());
        r.n = r.ndet[0];
        r.id = got;
        return got;
    }
}
