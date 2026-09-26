package org.onnxsim.maskrcnndemo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;

import java.io.File;

/**
 * Restarts an engine activity in a fresh process (its own, see the manifest): the Camera / Images
 * button. The engine activities end their process in onDestroy, so starting the same activity from
 * itself would land in the dying process; this waits (up to 3 s) for the caller's pid to be gone,
 * then starts the target intent ("target" class name, the rest of the extras passed on).
 */
public class RelaunchActivity extends Activity {
    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        Intent in = getIntent();
        int pid = in.getIntExtra("pid", -1);
        Intent out = new Intent();
        out.setClassName(this, in.getStringExtra("target"));
        if (in.getExtras() != null) out.putExtras(in.getExtras());
        out.removeExtra("target");
        out.removeExtra("pid");
        Handler h = new Handler(Looper.getMainLooper());
        long t0 = System.currentTimeMillis();
        Runnable poll = new Runnable() {
            @Override
            public void run() {
                if (pid > 0 && new File("/proc/" + pid).exists() && System.currentTimeMillis() - t0 < 3000) {
                    h.postDelayed(this, 50);
                    return;
                }
                startActivity(out);
                finish();
                overridePendingTransition(0, 0);
            }
        };
        h.postDelayed(poll, 50);
    }
}
