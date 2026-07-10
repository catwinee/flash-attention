# Copyright (c) 2026, Tri Dao.
"""Helpers for warp-specialized pipeline tracing.

The target design is SMEM-staged tracing: stamp ``%clock`` into shared memory at
synchronization boundaries, then flush the trace to global memory after the hot
pipeline has drained. Keeping the event path to ``clock + st.shared`` avoids the
per-stamp global store used by simpler timeline instrumentation.

The current FA4 fwd/bwd trace wiring intentionally starts with direct global
stores (``clock + st.global.u32``). That is more intrusive, but it gives a
complete dependency graph first. Full SMEM staging needs a more compact event
encoding or carefully reused slack space because SM100 fwd/bwd shared memory is
already near the launch limit.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


TRACE_MODE_NONE = 0
TRACE_MODE_MARKER = 1
TRACE_MODE_SYNC = 2
TRACE_MODE_FULL = 3

_TRACE_MODE = TRACE_MODE_FULL


def set_trace_mode(mode: int) -> None:
    """Set the compile-time trace event filter used by subsequent JIT compiles."""
    global _TRACE_MODE
    _TRACE_MODE = int(mode)


def normalize_trace_mode(mode: str | int) -> int:
    """Convert a user-facing trace mode into a small compile-key integer."""
    if isinstance(mode, str):
        normalized = mode.lower().replace("-", "_")
        if normalized in {"none", "off", "no_trace"}:
            return TRACE_MODE_NONE
        if normalized in {"marker", "markers", "marker_only"}:
            return TRACE_MODE_MARKER
        if normalized in {"sync", "sync_boundary", "sync_boundaries"}:
            return TRACE_MODE_SYNC
        if normalized in {"full", "detailed", "all"}:
            return TRACE_MODE_FULL
        raise ValueError(
            "pipeline trace mode must be one of: none, marker, sync, full"
        )
    mode = int(mode)
    if mode < TRACE_MODE_NONE or mode > TRACE_MODE_FULL:
        raise ValueError("pipeline trace mode must be in [0, 3]")
    return mode


def trace_mode_name(mode: int) -> str:
    return {
        TRACE_MODE_NONE: "none",
        TRACE_MODE_MARKER: "marker",
        TRACE_MODE_SYNC: "sync",
        TRACE_MODE_FULL: "full",
    }[int(mode)]


def _bwd_event_enabled(role: int, event: int) -> bool:
    """Compile-time event filter for the current SM100 backward trace map.

    The current bwd instrumentation only calls the generic stamp helper from the
    SM100 backward kernel. Keeping the filter here avoids touching every stamp
    call when comparing trace levels.
    """
    role = int(role)
    event = int(event)
    mode = _TRACE_MODE
    if mode == TRACE_MODE_NONE:
        return False
    if mode >= TRACE_MODE_FULL:
        return True
    marker_events = {
        0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),  # TMA issue markers.
        1: (1, 3, 5, 7, 9),  # MMA result-ready markers.
        2: (2, 5, 7, 14, 22, 25, 26),  # Compute result-ready / exchange markers.
        3: (3, 10, 13, 14),  # dQ reduce/store completion markers.
        4: (),
    }
    if mode == TRACE_MODE_MARKER:
        return event in marker_events.get(role, ())
    sync_events = {
        0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        # Producer-ready events plus wait enter/exit pairs.
        1: (
            1, 3, 5, 7, 9,
            10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31,
        ),
        # S and dP wait pairs, plus P and dS ready events.
        2: (
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 13, 14, 15, 16, 17, 18, 20, 21, 22,
            24, 25, 26, 27,
        ),
        # dQ wait pair plus reduce active bracket.
        3: (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 13, 14),
        4: (),
    }
    return event in sync_events.get(role, ())


@dsl_user_op
def clock_u32(*, loc=None, ip=None) -> cutlass.Uint32:
    """Read the per-SM 32-bit cycle counter."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %clock;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_smem_u32(
    smem_ptr: cute.Pointer,
    word_offset: Int32,
    value: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one u32 to shared memory at ``smem_ptr + 4 * word_offset``."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_ptr_i32,
            Int32(word_offset).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .u32 addr;\n\t"
        "shl.b32 addr, $1, 2;\n\t"
        "add.u32 addr, addr, $0;\n\t"
        "st.shared.u32 [addr], $2;\n\t"
        "}\n",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def load_smem_u32(
    smem_ptr: cute.Pointer,
    word_offset: Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Load one u32 from shared memory at ``smem_ptr + 4 * word_offset``."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr_i32,
                Int32(word_offset).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .u32 addr;\n\t"
            "shl.b32 addr, $2, 2;\n\t"
            "add.u32 addr, addr, $1;\n\t"
            "ld.shared.u32 $0, [addr];\n\t"
            "}\n",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_gmem_u32(
    gmem_ptr: cute.Pointer,
    word_offset: Int32,
    value: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one u32 to global memory at ``gmem_ptr + 4 * word_offset``."""
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            gmem_ptr_i64,
            Int32(word_offset).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .u64 addr;\n\t"
        ".reg .u64 byte_offset;\n\t"
        "mul.wide.u32 byte_offset, $1, 4;\n\t"
        "add.u64 addr, $0, byte_offset;\n\t"
        "st.global.u32 [addr], $2;\n\t"
        "}\n",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def flush_smem_to_gmem_u32x4(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    word_offset: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Copy four contiguous u32 trace words from shared memory to global memory."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_ptr_i32,
            gmem_ptr_i64,
            Int32(word_offset).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .u32 saddr;\n\t"
        ".reg .u64 gaddr;\n\t"
        ".reg .u64 byte_offset;\n\t"
        ".reg .b32 v0, v1, v2, v3;\n\t"
        "shl.b32 saddr, $2, 2;\n\t"
        "add.u32 saddr, saddr, $0;\n\t"
        "mul.wide.u32 byte_offset, $2, 4;\n\t"
        "add.u64 gaddr, $1, byte_offset;\n\t"
        "ld.shared.v4.u32 {v0, v1, v2, v3}, [saddr];\n\t"
        "st.global.v4.u32 [gaddr], {v0, v1, v2, v3};\n\t"
        "}\n",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def stamp_smem(
    trace_smem: Optional[cute.Tensor],
    role: cutlass.Constexpr[int],
    event: cutlass.Constexpr[int],
    iteration: Int32,
    n_events: cutlass.Constexpr[int],
    max_iterations: cutlass.Constexpr[int],
    enabled: Boolean | bool = True,
) -> None:
    """Stamp ``%clock`` into a shared-memory trace buffer.

    The logical layout matches the external timeline plotter:
    ``trace[(role * n_events + event) * max_iterations + iteration]``.
    Only one lane in each calling warp writes the stamp.
    """
    if const_expr(trace_smem is not None):
        if enabled and iteration < max_iterations:
            with cute.arch.elect_one():
                word_offset = Int32((role * n_events + event) * max_iterations) + iteration
                store_smem_u32(trace_smem.iterator, word_offset, clock_u32())


@cute.jit
def stamp_gmem(
    trace_gmem: Optional[cute.Tensor],
    role: cutlass.Constexpr[int],
    event: cutlass.Constexpr[int],
    iteration: Int32,
    n_events: cutlass.Constexpr[int],
    max_iterations: cutlass.Constexpr[int],
    enabled: Boolean | bool = True,
) -> None:
    """Stamp ``%clock`` directly into a global-memory trace buffer.

    This mirrors the reference profiler and is useful while establishing event
    semantics. It is currently what FA4 fwd/bwd use so the pipeline dependency
    graph can be validated before spending SMEM budget on a lower-overhead
    staging path. Prefer :func:`stamp_smem` once there is enough trace storage
    slack or a compact ring-buffer format.
    """
    if const_expr(trace_gmem is not None and _bwd_event_enabled(role, event)):
        if enabled and iteration < max_iterations:
            with cute.arch.elect_one():
                word_offset = Int32((role * n_events + event) * max_iterations) + iteration
                store_gmem_u32(trace_gmem.iterator, word_offset, clock_u32())


@cute.jit
def flush_smem_to_gmem(
    trace_smem: Optional[cute.Tensor],
    trace_gmem: Optional[cute.Tensor],
    n_words: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
) -> None:
    """Flush a u32 shared-memory trace buffer to global memory.

    Call this after all producer/consumer timeline events are complete and after
    a CTA-wide synchronization that makes shared stores visible to the flush
    threads. The hot event path does not perform global memory traffic.
    """
    if const_expr(trace_smem is not None and trace_gmem is not None):
        tidx = cute.arch.thread_idx()[0]
        n_vec_words = const_expr((n_words // 4) * 4)
        for word_offset in cutlass.range(tidx * 4, n_vec_words, num_threads * 4, unroll=1):
            flush_smem_to_gmem_u32x4(trace_smem.iterator, trace_gmem.iterator, word_offset)
        if const_expr(n_words % 4 != 0):
            for word_offset in cutlass.range(
                n_vec_words + tidx, n_words, num_threads, unroll=1
            ):
                value = load_smem_u32(trace_smem.iterator, word_offset)
                store_gmem_u32(trace_gmem.iterator, word_offset, value)


@cute.jit
def clear_smem(
    trace_smem: Optional[cute.Tensor],
    n_words: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
) -> None:
    """Zero a shared-memory trace buffer before recording events."""
    if const_expr(trace_smem is not None):
        tidx = cute.arch.thread_idx()[0]
        for word_offset in cutlass.range(tidx, n_words, num_threads, unroll=1):
            store_smem_u32(trace_smem.iterator, word_offset, cutlass.Uint32(0))
