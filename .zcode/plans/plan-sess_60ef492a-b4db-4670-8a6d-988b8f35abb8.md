## Context

Su Mac (48GB) lo startup con `--mixed-from-usage` resta silenzioso per minuti tra `speed: mx.compile applied to 80 FFN sub-calls` e le righe del warmup: il warmup (`engine.py:638`) paga la JIT del Metal al primo forward e non stampa nulla né prima né durante. Lo stesso vale per il load dei pesi, la passata mixed per-layer e `mx.compile`. Il flag `--verbose/-v` esiste già (`args.py:118`) ma è usato solo per `log_level` di uvicorn (`server.py:1147`); il log `.info` di `mixed_quantize.py:734` è invisibile perché nessuno configura `logging`.

Obiettivo: una riga **sempre** visibile prima del warmup (il caso "sembra bloccato"), e dettaglio per-fase/per-layer sotto `-v`.

## Design

Due livelli, nello stile esistente `_log` su stderr ("topic: message", `cli.py:41`):

1. **Sempre attivo** (non serve `-v`): prima di `engine.warmup()`, una riga che spiega che il primo forward paga la compilazione JIT e può richiedere minuti. È la riga che mancava nel tuo caso.
2. **Sotto `-v`**: per-file/per-layer/per-fase con timing; più `logging.basicConfig(INFO)` così il `.info` esistente di mixed_quantize e i warning dei moduli compaiono.

Messaggi in inglese (stile домinante), callback passate come parametri opzionali `log`/`progress` con default `None` → nessun cambio comportamento per default e per il comando `mixed-quantize`.

## Modifiche

- **`src/agrimlxo/cli.py`** (`cmd_serve`, ~309-477):
  - se `args.verbose`: `logging.basicConfig(level=INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")` e `vlog = _log`; altrimenti `vlog = None` (no-op).
  - timing `time.perf_counter()` per fase; riga finale `startup: ready in Xs (load Xs, mixed Xs, warmup Xs)`.
  - `LoadedModel(..., log=vlog)`, `apply_usage_mix_*(..., progress=vlog)`, `engine.warmup(progress=vlog)`.
  - **sempre**: `_log("speed: warmup — first forward pays JIT kernel compilation (can take minutes)")` prima della chiamata; le righe post-warmup esistenti restano.
- **`src/agrimlxo/loader.py`** (`LoadedModel.__init__`, 133-181): parametro `log=None`; per-file `load: weights file i/N` nel loop `:169-170` e righe di fase sanitize/quantize/load_weights quando `log` è dato.
- **`src/agrimlxo/mixed_quantize.py`**: parametro `progress=None` in `apply_usage_mix_in_memory` (`:391`) e `apply_usage_mix_4tier` (`:667`): per-layer `mixed 4-tier: layer k/N (cache hits X, misses Y)` prima di `mx.eval(sm.parameters())` (`:674`, il passo lento). Riutilizza i contatori già presenti `ExpertQuantCache.hits/misses` (`:77-78`).
- **`src/agrimlxo/engine.py`** (`warmup`, 638-713): parametro `progress=None` (callable); righe a confine fase (prefill, greedy decode, sampled, memory) e dentro i loop decode un battito throttled (~ogni 2s, stesso pattern del throttle 0.25s di `engine.py:498-515`): `warmup: decode 12/32`. Attenzione all'invariante pinnato: le righe devono restare fuori da `rt.maybe_print_stats()` e non toccare `_stats` (test `tests/test_speed.py:36-59`).
- **`src/agrimlxo/args.py`**: help di `-v` aggiornato: "verbose startup logging (phases, per-layer progress) + uvicorn info".
- **`README.md`**: menzione di `-v` nella riga "Server options".

## Verifica

- `pytest tests/` (suite CPU-compatibile, 61 test): i pin esistenti su stderr (`test_args.py:132`, `test_engine_toy.py:257`) non devono cambiare — le nuove righe sono dietro callback o verbose.
- Nuovo mini-test in `tests/test_speed.py`: `warmup(progress=collect)` su toy model raccoglie le fasi attese.
- Su Mac (manuale, come da protocollo): `agrimlxo serve -m mlx-community/Qwen3.5-35B-A3B-8bit --mixed-from-usage --no-moe-expansion --mem-target 30GB -v` → si vede avanzamento per-layer della passata mixed e il battito `warmup: decode k/32` durante la JIT; senza `-v` compare solo la nuova riga pre-warmup.