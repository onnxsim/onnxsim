#!/usr/bin/env python3
"""System-level profile of NPU runs on an Android Allwinner device, from a Perfetto trace taken while the workload runs.

    pip install perfetto
    trace_npu.py --serial SERIAL --nb yolov5s_rt_uint8_a733.nb [--iters 100] [--taskset 80]

VIPLite reports whole-network NPU time and cycles (`viplite_hw`), nothing per layer. What the adb shell *can* see from outside is
recorded here with Perfetto (the `perfetto` service enables ftrace events for the shell user; /sys/kernel/tracing is read-only):

* the NPU's interrupt (`vipcore_0`): one per inference, so its count cross-checks the submissions and its spacing is the call period;
* the NPU temperature (`npu_thermal_zone`), which the sysfs thermal nodes do not show to the shell;
* the worker thread's scheduling: how much of each call it is on a CPU (conversion + submit) versus asleep waiting for the NPU, the
  wake-up latency after the completion interrupt, and which core class it ran on (little cores make the float32 conversion slower);
* devfreq/clk events for the NPU clock (none appear on the A733: the clock is not changed through the Linux clock framework).

Per-layer timing, DRAM traffic and utilization are not observable this way. Tracing slows the CPU-side conversion a little (the
NPU time itself is unchanged), so use bench.py for timings and this for the system view.
"""

import argparse
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

CONFIG = """
buffers {{ size_kb: 131072 fill_policy: DISCARD }}
data_sources {{ config {{ name: "linux.ftrace" ftrace_config {{
  ftrace_events: "sched/sched_switch"
  ftrace_events: "sched/sched_waking"
  ftrace_events: "irq/irq_handler_entry"
  ftrace_events: "irq/irq_handler_exit"
  ftrace_events: "power/cpu_frequency"
  ftrace_events: "power/clock_set_rate"
  ftrace_events: "clk/clk_set_rate"
  ftrace_events: "devfreq/devfreq_frequency"
  ftrace_events: "thermal/thermal_temperature"
}} }} }}
duration_ms: {duration_ms}
"""


def adb(serial, command, stdin=None, timeout=300):
    return subprocess.run(
        ["adb", "-s", serial, "shell", command],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def capture(a, out):
    remote = "/data/misc/perfetto-traces/aw_trace.pftrace"
    cfg = CONFIG.format(duration_ms=a.duration_ms)
    # The config goes over stdin: perfetto cannot read /data/local/tmp from the shell's SELinux domain.
    adb(a.serial, f"perfetto --background --txt -c - -o {remote}", stdin=cfg)
    time.sleep(1.5)
    pin = f"taskset {a.taskset} " if a.taskset else ""
    run = adb(
        a.serial,
        f"cd {a.dir} && {pin}./onnx-remote-viplite-worker --bench {a.nb} {a.iters} 2>&1",
        timeout=600,
    )
    print(run.stdout.strip(), file=sys.stderr)
    time.sleep(max(0.0, a.duration_ms / 1000 - 3))
    for _ in range(40):  # wait for perfetto to flush and exit
        if not adb(a.serial, "pidof perfetto").stdout.strip():
            break
        time.sleep(0.5)
    subprocess.run(
        ["adb", "-s", a.serial, "pull", remote, str(out)],
        check=True,
        capture_output=True,
    )
    return run.stdout


def core_capacity(serial):
    out = adb(
        serial,
        "for c in 0 1 2 3 4 5 6 7; do echo $c $(cat /sys/devices/system/cpu/cpu$c/cpu_capacity 2>/dev/null); done",
    ).stdout
    return {
        int(c): int(v)
        for c, v in (
            line.split() for line in out.splitlines() if len(line.split()) == 2
        )
    }


def analyze(trace, capacity):
    from perfetto.trace_processor import TraceProcessor

    tp = TraceProcessor(trace=str(trace))
    q = lambda sql: list(tp.query(sql))  # noqa: E731
    lines = []
    irqs = q("select ts, dur from slice where name like 'IRQ (vipcore%' order by ts")
    if not irqs:
        return [
            "No `vipcore` interrupts in the trace: the workload did not overlap the capture window."
        ]
    ts = [r.ts for r in irqs]
    gaps = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
    lines.append(
        f"- NPU interrupts: {len(irqs)} (one per inference); handler median {statistics.median(r.dur for r in irqs) / 1e3:.1f} us"
    )
    lines.append(
        f"- call period between completions: median {statistics.median(gaps):.1f} ms, min {min(gaps):.1f}, max {max(gaps):.1f}"
    )

    th = q(
        "select c.ts, c.value from counter c join counter_track t on c.track_id=t.id where t.name like 'npu%thermal%' order by c.ts"
    )
    if th:
        v = [r.value / 1000 for r in th]
        lines.append(
            f"- NPU temperature: {v[0]:.1f} -> max {max(v):.1f} C over the window ({len(v)} samples)"
        )
    for name in ("devfreq", "clk_set_rate", "clock_set_rate"):
        n = q(f"select count(*) c from ftrace_event where name like '%{name}%'")[0].c
        if n:
            lines.append(
                f"- {n} `{name}` events (the clock was changed during the run)"
            )

    worker = q(
        "select utid, tid from thread where name like 'onnx-remote-vi%' order by tid desc limit 1"
    )
    if worker:
        utid = worker[0].utid
        runs = q(f"select ts, dur, cpu from sched where utid={utid} order by ts")
        on, wake, cpus = [], [], []
        for a, b in zip(ts, ts[1:]):
            on.append(
                sum(max(0, min(r.ts + r.dur, b) - max(r.ts, a)) for r in runs) / 1e6
            )
            nxt = next((r for r in runs if r.ts >= a), None)
            if nxt:
                wake.append((nxt.ts - a) / 1e3)
                cpus.append(nxt.cpu)
        period = statistics.median(gaps)
        lines.append(
            f"- worker on a CPU {statistics.median(on):.1f} ms of each {period:.1f} ms call ({100 * statistics.median(on) / period:.0f}%); "
            "asleep while the NPU runs (no spin-wait)"
        )
        if wake:
            lines.append(
                f"- completion interrupt -> worker running again: median {statistics.median(wake):.0f} us, p90 {sorted(wake)[int(0.9 * len(wake))]:.0f}, max {max(wake):.0f}"
            )
        by = Counter()
        for r in runs:
            by[r.cpu] += r.dur / 1e6
        total = sum(by.values()) or 1
        little = sum(v for c, v in by.items() if capacity.get(c, 1024) < 1024)
        place = ", ".join(
            f"cpu{c} {100 * v / total:.0f}%" for c, v in by.most_common(3)
        )
        lines.append(
            f"- worker CPU time by core: {place}; {100 * little / total:.0f}% on little cores (capacity {sorted(set(capacity.values()))})"
        )
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--serial", required=True)
    p.add_argument("--dir", default="/data/local/tmp/viplite")
    p.add_argument("--nb", required=True, help="network inside --dir")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--duration-ms", type=int, default=9000)
    p.add_argument(
        "--taskset",
        help="hex CPU mask for the worker, e.g. 80 = cpu7 (a big core on the A733)",
    )
    p.add_argument(
        "--trace",
        default="npu_trace.pftrace",
        help="where to keep the pulled trace (open it in ui.perfetto.dev)",
    )
    a = p.parse_args(argv)
    capture(a, Path(a.trace))
    print("\n".join(analyze(Path(a.trace), core_capacity(a.serial))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
