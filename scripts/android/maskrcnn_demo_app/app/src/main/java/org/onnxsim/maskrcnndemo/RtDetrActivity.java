package org.onnxsim.maskrcnndemo;

import android.graphics.Bitmap;
import android.media.Image;

import java.io.File;
import java.util.Locale;

/**
 * RT-DETR-r18 mode (its own process, see the manifest): the YOLO mode's camera/images loop with
 * RT-DETR (../vision_models/rtdetr, PR #1867) as the detector -- NMS-free, 4 HTP pieces around 3
 * deformable-attention calls on the HVX (native/rtdetr_engine.cpp). Extra "opts":
 * "flags=260;thresh=0.4" (MSDA threads/jobs, score threshold).
 */
public class RtDetrActivity extends YoloActivity {
    @Override
    String activityKey() {
        return "rtdetr";
    }

    @Override
    int cameraMinWidth() {
        return RtDetrEngine.IN;
    }

    @Override
    String defaultModel() {
        return "rtdetr-r18";
    }

    @Override
    void switchInPlace(String model) {}

    @Override
    String engineInit(String model, String opts) {
        return RtDetrEngine.nativeInit(new File(getFilesDir(), "models").getAbsolutePath(),
                getApplicationInfo().nativeLibraryDir, opts);
    }

    @Override
    Engine.Result newResult(String model) {
        return new Engine.Result(RtDetrEngine.MAX_DET, false, RtDetrEngine.T_N, 0f);  // the engine thresholds
    }

    @Override
    int[] fitDims(int w, int h, int rot) {
        int[] d = new int[2];
        RtDetrEngine.nativeFitDims(w, h, rot, d);
        return d;
    }

    @Override
    void runYuv(Image img, int rot, Bitmap disp, Engine.Result r) {
        Image.Plane[] p = img.getPlanes();
        int n = RtDetrEngine.nativeRunYuv(p[0].getBuffer(), p[1].getBuffer(), p[2].getBuffer(), p[0].getRowStride(),
                p[1].getRowStride(), p[1].getPixelStride(), img.getWidth(), img.getHeight(), rot, disp, r.boxes,
                r.labels, r.scores, r.times);
        if (n < 0) throw new RuntimeException(RtDetrEngine.nativeLastError());
        r.n = n;
    }

    @Override
    void run(Bitmap rgba, Engine.Result r) {
        int n = RtDetrEngine.nativeRun(rgba, r.boxes, r.labels, r.scores, r.times);
        if (n < 0) throw new RuntimeException(RtDetrEngine.nativeLastError());
        r.n = n;
    }

    @Override
    String stages(double[] avg) {
        return String.format(Locale.US, "pre %.1f  htp %.1f  msda %.1f (dsp %.1f)  post %.2f ms", avg[RtDetrEngine.T_PRE],
                avg[RtDetrEngine.T_HTP], avg[RtDetrEngine.T_MSDA], avg[RtDetrEngine.T_MSDA_DSP], avg[RtDetrEngine.T_POST]);
    }
}
