# AgrimlxO — First run guide (Mac)

Step-by-step for every unified-memory tier: 16 / 32 / 48 / 64 GB.

---

## 1. Install (once)

```bash
git clone https://github.com/vagrillo/AgrimlxO && cd AgrimlxO
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q        # 92 tests — must be green on your Mac too
```

---

## 2. Pick the model for your RAM

The reference model is **Qwen3.5-35B-A3B (MoE, 256 experts, native top-8)**.

| Mac RAM | checkpoint | command |
|---|---|---|
| **16 GB** | 2–3 bit MLX (search HF: `mlx-community`/`ISTA-DASLab`) | `agrimlxo download <repo-id>` |
| **32 GB** | 4-bit MLX | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-4bit` |
|   oppure | 6-bit (vedi §4 lazy) | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-6bit` |
|   or | 6-bit (see §4 lazy) | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-6bit` |
| **48 GB** | 6-bit MLX (the reference) | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-6bit` |
|   or | 8-bit | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-8bit` |
| **64 GB** | 8-bit MLX | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-8bit` |

If no MLX quant exists on HF yet, generate one from the bf16 source (see §6).

---

## 3. First boot

```bash
# the expansion is OFF by default — add --moe-expansion to enable it:
# stock (top-8 native, no expansion — the default):
agrimlxo serve -m <modello> --host 0.0.0.0

# with expansion (opt-in via --moe-expansion):
agrimlxo serve -m <modello> --moe-expansion --host 0.0.0.0

# expected stderr with expansion:
#   moe: default expansion profile injected (Qwen3.5/3.6-35B-A3B): --moe-experts 12 ...
#   moe: routed experts per token: 12 (model default 8; ranks 9..12 get linear influence decay 0.99..0.5)
#   moe: expert threshold: keep experts while p >= 0.75 x p(rank 6) (experts 3..12 per token)
#   moe: expansion active on layers 25..39 of 40
#   speed: mx.compile applied to 80 FFN sub-calls
#   speed: warmup decode — greedy <G> tok/s | sampled <S> tok/s
#   listening on http://0.0.0.0:8080  (slots: 1, ctx: 131072)
```

Open **http://127.0.0.1:8080** for the chat interface (or use any OpenAI client).

Expected speed on M-series (6-bit): decode ~50–55 tok/s with the expansion
profile, ~65 stock. First response is slower (Metal JIT) — the warmup pays
that cost at startup.

---

## 4. 16 GB Macs: the lazy mixed-quant workflow

With 16 GB the 6-bit checkpoint (~28 GB) doesn't fit — but the **lazy
per-layer loader** processes one layer at a time, so the peak is the MIXED
size (~12 GB), not the source:

```bash
agrimlxo serve \
    -m mlx-community/Qwen3.5-35B-A3B-6bit \
    --mixed-from-usage --mem-target 12GB \
    --save-mixed ~/models/qwen36-mixed \
    --no-moe-expansion \
    --host 0.0.0.0 --port 7080
```

What happens:
1. the 6-bit is loaded **one layer at a time** (lazy from safetensors),
2. each layer's routed experts are dequantized and re-quantized into
   4 tiers (8/6/4/2 bit) per the usage ranking, within the 12 GB target,
3. the mixed model is **saved to `~/models/qwen36-mixed/`** — the next
   boots load it directly (no re-mix, no 28 GB peak):

```bash
agrimlxo serve -m ~/models/qwen36-mixed --host 0.0.0.0 --port 7080
```

> **Note:** the first run still needs to READ the 28 GB checkpoint from
> disk (streamed, not held in memory). On a 16 GB Mac the peak is the
> mixed model + one source layer (~13 GB) — it fits.

The USAGE heatmap (`USAGE` button in the header) shows the LED matrix that
fed the ranking; **Quant map** shows the assigned bit-widths (red 8 /
orange 6 / yellow 4 / white 2). After some conversations, re-run with a
fresh ranking to re-balance the tiers.

---

## 5. 32 / 48 / 64 GB: the same flow, bigger tiers

The `--mem-target` solver distributes 8/6/4/2-bit tiers hot-first within
the budget — more GB = more experts at higher bits:

```bash
# 32 GB: 4-bit base, hot experts at 6–8 bit
agrimlxo serve -m <model> --mixed-from-usage --mem-target 20GB \
    --save-mixed ~/models/qwen36-mixed --host 0.0.0.0

# 48 GB: 6-bit base (the reference), hot experts at 8-bit
agrimlxo serve -m <model> --mixed-from-usage --mem-target 28GB \
    --save-mixed ~/models/qwen36-mixed --host 0.0.0.0

# 64 GB: everything at 8-bit is affordable without mixing
agrimlxo serve -m <model> --host 0.0.0.0
```

Monitor the actual memory: the stderr line
`mixed 4-tier: memoria modello <N> MB (target …)` — and `ps -o rss=`.

---

## 6. If no MLX quant exists: make one from bf16

```bash
agrimlxo download Qwen/Qwen3.5-35B-A3B          # bf16 source (~70 GB disk)
agrimlxo mixed-quantize ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*/ \
    --top-m 256 --hi-bits 8 --lo-bits 4 --out ~/models/qwen36-8bit
```

(For the lazy path you don't need this: `--mixed-from-usage` works directly
on the bf16 checkpoint, processing one layer at a time.)

---

## 7. Tools, kache, and the extras

```bash
# tools (the model can call local python scripts in ./tools/):
agrimlxo serve -m <m> --tools

# persistent conversation KV snapshots (survive a restart):
agrimlxo serve -m <m> --kache --kache-budget 8GB

# disable reasoning (direct answers):
agrimlxo serve -m <m> --no-thinking

# no expansion at all (stock top-8, bit-identical):
agrimlxo serve -m <m> --no-moe-expansion
```

The webui extras: 📎 attach images/PDFs (PDF as text or page images at a
configurable DPI), 🔄 regenerate an answer, USAGE heatmap (LED matrix +
per-layer quant map), MOE panel (change the expansion at runtime), chat
history (client-side).

---

## 8. Troubleshooting

| symptom | fix |
|---|---|
| `ModuleNotFoundError: agrimlxo.models` | `git pull && pip install -e .` |
| `SwitchLinear.to_quantized() got unexpected 'mode'` | fixed in 6320d98 — `git pull` |
| `gather_qmm ... global_scale` | fixed in e0e2ddf — `git pull` |
| detokenizer garbled text | fixed in 4cb7783 — `git pull` |
| swap / memory pressure | Activity Monitor → "Memory used" + "Swap used"; lower `--mem-target` or use a smaller quant |
| first response very slow | warmup pays the Metal JIT at startup — check the `speed: warmup` line |
| everything looks 8-bit in Quant map on a 6-bit model | fixed in 2d97bd0 — `git pull` |

## 9. Quality measurement (paired protocol)

Same prompts, `--temp 0`, ONE variable per run: expansion OFF vs ON. The
paper's protocol (README → Zenodo link) measures on GPQA-Diamond; on
Qwen3.6-35B-A3B the expansion measured +2.52 pts (84.34% vs 81.82%).
