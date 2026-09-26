/*
 * SPDX-License-Identifier: Apache-2.0
 */

// A lightweight profiler for onnxsim's simplification fixed-point functions.
//
// Simplification runs a handful of transforms -- shape inference, the
// onnx-optimizer passes, constant folding and (optionally) a graph rewriter --
// alternately, to a joint fixed point (see ``FixedPointFn`` in onnxsim.cpp).
// This profiler measures the wall-clock and CPU duration of each of those
// functions, and the peak resident-set size (RSS) reached while each runs, then
// writes the result as a Chrome Trace Event Format JSON. That file opens
// directly in the Chrome tracing visualizer (``chrome://tracing``) or the
// Perfetto UI (https://ui.perfetto.dev) and renders as a flame graph: the
// nested fixed points appear as parent spans and the individual transforms as
// their children, one box per invocation. A concise per-function summary is
// also printed to stdout.
//
// Profiling is opt-in and adds no overhead when disabled. onnxsim.cpp turns it
// on when the ``ONNXSIM_PROFILE`` environment variable names an output file
// (mirroring the existing ``ONNXSIM_FIXED_POINT_ITERS`` knob), so it works
// unchanged from every binding -- Python, the C ABI and the Rust wrapper.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace onnxsim {

// Records nested timing/memory spans and emits a Chrome trace + summary.
//
// The profiler is a process-wide singleton. Spans are pushed/popped per thread
// (a thread-local stack), so a transform that spins up worker threads still
// nests correctly, and the shared event/sample buffers are mutex-guarded. While
// profiling is active a background thread samples RSS at a fixed interval so
// each span can report the peak memory observed during its lifetime, not merely
// the value at its boundaries.
class Profiler {
 public:
  static Profiler& Instance();

  // Turn profiling on, writing the trace to ``out_path`` when Finish() runs.
  // Resets any state from a previous run and starts the RSS sampler. Calling
  // Enable() again (e.g. onnxsim's large-model fallback simplifies twice)
  // starts a fresh run whose trace overwrites the previous one.
  void Enable(const std::string& out_path);

  bool enabled() const { return enabled_; }

  // Push / pop a named span on the calling thread's stack. Both are no-ops when
  // profiling is disabled, so the wrappers in onnxsim.cpp cost nothing then.
  void Begin(const std::string& name);
  void End();

  // Stop sampling, write the trace file and print the summary. A no-op when
  // profiling was never enabled. After it returns profiling is off again.
  void Finish();

  // Merge ONNX Runtime's own per-session profiling traces into this trace.
  //
  // When enabled, Finish() reads each trace path handed to AddOrtTracePath()
  // and splices its per-operator events under the matching ``OrtSession`` span
  // (offsetting ONNX Runtime's session-relative timestamps into this profiler's
  // timeline, one ONNX Runtime thread per track), then deletes the file. This
  // is the C++ (binding-agnostic) counterpart of onnxsim's Python
  // ``merge_ort_profile`` and is what lets the C ABI, Rust and WASM bindings
  // produce a single unified trace. Driven by ``ONNXSIM_MERGE_ORT_PROFILE``.
  void SetMergeOrtTraces(bool on) { merge_ort_traces_ = on; }
  bool merge_ort_traces() const { return merge_ort_traces_; }

  // Record the path of an ONNX Runtime trace to merge at Finish(). A no-op
  // unless profiling and merging are both on.
  void AddOrtTracePath(const std::string& path);

  // RSS sampling interval, milliseconds. Read from ONNXSIM_PROFILE_INTERVAL_MS
  // by Enable() (default 5ms); exposed for tests.
  void set_sample_interval_ms(unsigned ms);

  // Record a live node-count sample for the "node reduction per loop" curve:
  // how many nodes the graph holds right after one round of a fixed-point
  // loop (``loop`` names which one, e.g. "Optimize", "FoldConstant",
  // "Rewrite", or "Initial" for the count before simplification starts).
  // Written to the trace as "NodeCount" counter (``ph:"C"``) events tagged
  // with ``loop`` in their args, so onnxsim/profile_plot.py can group them
  // into one subplot per loop. A no-op unless profiling is enabled -- callers
  // should still guard the (O(n)) node count they pass in behind enabled()
  // themselves, so that count is never computed when profiling is off.
  void RecordNodeCount(const std::string& loop, size_t node_count);

  // Add an event produced by a remote executor or accelerator. ``ts_us`` is
  // in this profiler's timeline, and ``args_json`` must be a JSON object (or
  // empty for no arguments). This is intentionally separate from ProfiledScope
  // because the event may have been measured by another process.
  void RecordExternalEvent(const std::string& name, const std::string& category,
                           uint64_t ts_us, uint64_t duration_us,
                           const std::string& args_json = "{}");

  // Elapsed wall-clock time since Enable(), in microseconds. Remote clients
  // use this to anchor worker-relative timestamps in the local trace.
  uint64_t ElapsedMicros() const;

 private:
  Profiler() = default;
  ~Profiler();
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  struct Impl;
  // Heap-allocated so this header stays free of <thread>/<mutex>/<vector>.
  Impl* impl_ = nullptr;
  bool enabled_ = false;
  bool merge_ort_traces_ = false;
};

// RAII wrapper that profiles the enclosing scope. Captures whether profiling is
// enabled at construction so the matching End() is only emitted for a Begin()
// that actually happened.
class ProfiledScope {
 public:
  explicit ProfiledScope(const char* name)
      : active_(Profiler::Instance().enabled()) {
    if (active_) {
      Profiler::Instance().Begin(name);
    }
  }
  ~ProfiledScope() {
    if (active_) {
      Profiler::Instance().End();
    }
  }
  ProfiledScope(const ProfiledScope&) = delete;
  ProfiledScope& operator=(const ProfiledScope&) = delete;

 private:
  bool active_;
};

}  // namespace onnxsim
