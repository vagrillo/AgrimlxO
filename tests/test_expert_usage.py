# SPDX-License-Identifier: Apache-2.0
"""ExpertUsage accumulator tests: native top-K only, session/all scopes."""

import mlx.core as mx
import numpy as np

from agrimlxo.moe_expansion import (
    ExpertUsage,
    observe_usage,
    set_usage_sink,
)


def test_native_top_k_only_and_scopes():
    u = ExpertUsage(n_layer=4, n_expert=8)
    set_usage_sink(u)
    try:
        # espansione attiva: N=6 rank ma contano solo i primi 2 (top-K nativo)
        ids = mx.array([[[3, 5, 1, 7, 0, 2]]])  # rank-ordered
        observe_usage(2, ids, k_native=2)
        u.drain()
        # solo gli esperti 3 e 5 contano sul layer 2
        assert u.counts_all[2, 3] == 1 and u.counts_all[2, 5] == 1
        assert u.counts_all[2].sum() == 2
        assert u.counts_all[0].sum() == 0

        observe_usage(2, mx.array([[[3, 5, 1, 7, 0, 2]]]), k_native=2)
        u.drain()
        assert u.counts_all[2, 3] == 2
        # sessione rispecchia finché non si resetta
        assert u.counts_session[2, 3] == 2
        u.reset_session()
        assert u.counts_session[2, 3] == 0
        assert u.counts_all[2, 3] == 2  # all-time intatto

        snap = u.snapshot(session=True)
        assert snap["counts"][2][3] == 0 and snap["tokens"] == 0
        snap_all = u.snapshot()
        assert snap_all["counts"][2][3] == 2 and snap_all["tokens"] == 2
    finally:
        set_usage_sink(None)


def test_reset_clears_pending_lazy():
    u = ExpertUsage(n_layer=2, n_expert=4)
    set_usage_sink(u)
    try:
        lazy = mx.argmax(mx.zeros((1, 1, 4)), axis=-1)[..., None]
        observe_usage(0, lazy, k_native=1)
        observe_usage(0, lazy, k_native=1)
        u.reset_session()          # reset con entry ancora in coda
        u.drain()                  # il drain post-reset non deve contare
        assert u.counts_session[0].sum() == 0
        assert u.counts_all[0].sum() == 0
    finally:
        set_usage_sink(None)


def test_lazy_drain_defers_until_materialized():
    u = ExpertUsage(n_layer=2, n_expert=4)
    set_usage_sink(u)
    try:
        # array lazy in un grafo non valutato: observe non forza eval
        lazy = mx.argmax(mx.zeros((1, 1, 4)), axis=-1)[..., None]
        observe_usage(1, lazy, k_native=1)
        # il drain converte quando materializzato (mx.zeros è già concreto)
        u.drain()
        assert u.counts_all[1, 0] == 1
    finally:
        set_usage_sink(None)
