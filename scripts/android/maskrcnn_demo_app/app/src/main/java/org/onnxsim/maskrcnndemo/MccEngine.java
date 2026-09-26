package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;

/** JNI wrapper around native/mcc_engine.cpp (SAM tap -> MoGe-2 -> MCC, all on the HTP). */
final class MccEngine {
    /** Working image: portrait W x H = 480 x 640, landscape 640 x 480 (MoGe-2's two static shapes). */
    static final int SHORT = 480, LONG = 640;

    static {
        System.loadLibrary("mcc_demo");
    }

    /** Loads SAM + MCC (EP-context models compiled on the first launch; MoGe-2 on first use). null = ok. */
    static native String nativeInit(String modelDir, String nativeLibDir, String opts);
    /** Camera frame -> disp (the upright working bitmap); with encode it becomes the working image + SAM encoder. */
    static native boolean nativeYuv(java.nio.ByteBuffer y, java.nio.ByteBuffer u, java.nio.ByteBuffer v, int yRowStride,
                                    int uvRowStride, int uvPixelStride, int w, int h, int rot, Bitmap disp,
                                    boolean encode, float[] times);
    /** Upright RGBA working bitmap -> working image + SAM encoder. times[0]: encoder ms. */
    static native boolean nativeEncode(Bitmap rgba, float[] times);
    /** Tap (working pixels) -> mask (W x H, 1 inside), iou[4]; returns the SAM slot, -1 on error. */
    static native int nativeSegment(float x, float y, byte[] mask, float[] iou, float[] times);
    /**
     * MoGe-2 + MCC on the current mask. times: MoGe-2 run (it starts in the background when the image
     * is encoded), waited for it, prep, encoder, decoder, total ms; counts: queries, decoder chunks,
     * points. Returns the point count, -1 on error.
     */
    static native int nativeReconstruct(float[] times, int[] counts);
    /** The last reconstruction's points: xyz (3 per point) and ARGB colors. */
    static native void nativePoints(float[] xyz, int[] argb);
    static native String nativeLastError();

    /** Working size for an upright w x h frame: portrait or landscape. */
    static int[] workDims(int w, int h) {
        return h > w ? new int[] {SHORT, LONG} : new int[] {LONG, SHORT};
    }

    /** Center-crop an upright bitmap to 3:4 / 4:3 and scale it to the working size. */
    static Bitmap working(Bitmap src) {
        int[] d = workDims(src.getWidth(), src.getHeight());
        float r = Math.min((float) src.getWidth() / d[0], (float) src.getHeight() / d[1]);
        int cw = Math.round(d[0] * r), ch = Math.round(d[1] * r);
        Bitmap c = Bitmap.createBitmap(src, (src.getWidth() - cw) / 2, (src.getHeight() - ch) / 2, cw, ch);
        Bitmap out = Bitmap.createScaledBitmap(c, d[0], d[1], true);
        return out.getConfig() == Bitmap.Config.ARGB_8888 ? out : out.copy(Bitmap.Config.ARGB_8888, false);
    }

    static void check(boolean ok) {
        if (!ok) throw new RuntimeException(nativeLastError());
    }
}
