// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared between `split_large_gather` (which stamps this marker into the
// `doc_string` of every `Split` node it creates) and
// `fuse_split_gather_concat` (which declines to re-fuse a
// `Split`/`Gather`.../`Concat` group whose `Split` carries it): a `Split`
// deliberately introduced to keep each `Gather`'s indices tensor within a
// size-limited backend's limit must survive the latter, default-on pass, or
// the two would perpetually undo each other -- `split_large_gather`
// re-splitting what `fuse_split_gather_concat` just fused back together,
// forever, since one is opt-in (runs every simplification round it's asked
// for) and the other runs by default (every round, unconditionally).
//
// This rides in `doc_string` rather than as a real node attribute:
// `Split`'s ONNX schema does not declare any attribute by this name, so
// onnx's own attribute validation (run as part of shape inference on every
// simplification round) would reject the node outright the moment it tried
// to add one. `doc_string` is a plain free-text field every NodeProto
// carries regardless of op type, so it round-trips through the
// Graph<->ModelProto conversion the same way without tripping schema
// validation.
inline constexpr char kSizeLimitedGatherSplitMarker[] =
    "onnxsim_size_limited_gather_split";
