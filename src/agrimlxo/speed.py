# SPDX-License-Identifier: Apache-2.0
"""Speed: mx.compile of the pure FFN sub-calls and a startup warmup pass.

Two decode-speed levers the reference engines (mlx-lm/omlx) use that the
engine lacked:

1. ``compile_ffn(model)`` wraps the *pure* FFN sub-calls (SwitchGLU /
   MixedSwitchGLU, shared expert) in ``mx.compile``. These carry the bulk
   of the routed compute in an A3B model, have no cache side effects (the
   attention/GDN paths mutate their caches from Python, which mx.compile
   would freeze after tracing, so they stay uncompiled), and benefit from
   op fusion / reduced dispatch on Metal. Outputs are unchanged (same ops,
   fused); pinned by tests.

   mx.compile does not bind methods (``self`` would shift into the argument
   list), so each instance's bound call is compiled once and dispatched by
   ``maybe_compiled`` at the call sites; unregistered instances keep the
   stock path.

2. ``Engine.warmup()`` runs a short dummy generation before the server
   listens: pays the Metal kernel JIT / autotune / first-trace costs up
   front instead of on the first user request (the "first response much
   slower" artifact) and pre-traces the compiled variants for the prefill
   and decode shapes. MoE expansion observability is suspended so the
   stats window stays clean.
"""

import mlx.core as mx

# id(instance) -> (instance, compiled bound call). Keeping the instance in
# the value keeps id() unique for the process lifetime (single model).
_REGISTRY = {}


def maybe_compiled(obj, *args, **kwargs):
    """Call ``obj`` through its compiled implementation when registered,
    else the stock path."""
    entry = _REGISTRY.get(id(obj))
    if entry is None:
        return obj(*args, **kwargs)
    return entry[1](*args, **kwargs)


def compile_ffn(model) -> int:
    """mx.compile the pure FFN sub-calls of a loaded model in place.

    Returns the number of newly compiled callables (0 when already done).
    """
    n = 0
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        switch = getattr(mlp, "switch_mlp", None)
        if switch is not None and id(switch) not in _REGISTRY:
            _REGISTRY[id(switch)] = (switch, mx.compile(switch.__call__))
            n += 1
        shared = getattr(mlp, "shared_expert", None)
        if shared is not None and id(shared) not in _REGISTRY:
            _REGISTRY[id(shared)] = (shared, mx.compile(shared.__call__))
            n += 1
    return n
