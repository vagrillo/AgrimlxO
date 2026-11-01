# AgrimlxO — Guida al primo avvio (Mac)

Passo-passo per ogni taglio di memoria unificata: 16 / 32 / 48 / 64 GB.

---

## 1. Installazione (una volta)

```bash
git clone https://github.com/vagrillo/AgrimlxO && cd AgrimlxO
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q        # 92 test — devono essere verdi anche sul Mac
```

---

## 2. Scegli il modello per la tua RAM

Il modello di riferimento è **Qwen3.5-35B-A3B (MoE, 256 esperti, top-8 nativo)**.

| RAM Mac | checkpoint | comando |
|---|---|---|
| **16 GB** | 2–3 bit MLX (cerca su HF: `mlx-community`/`ISTA-DASLab`) | `agrimlxo download <repo-id>` |
| **32 GB** | 4-bit MLX | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-4bit` |
| **48 GB** | 6-bit MLX (il riferimento) | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-6bit` |
| **64 GB** | 8-bit MLX | `agrimlxo download mlx-community/Qwen3.5-35B-A3B-8bit` |

Se su HF non esiste ancora il quant MLX, generane uno dal bf16 (vedi §6).

---

## 3. Primo avvio

```bash
# il profilo default (esperti 12, soglia 0.75, layer 25–39) viene iniettato
# automaticamente per Qwen3.5/3.6-35B-A3B — lo vedi su stderr:
agrimlxo serve -m <modello> --host 0.0.0.0

# stderr atteso:
#   moe: default expansion profile injected (Qwen3.5/3.6-35B-A3B): --moe-experts 12 ...
#   moe: routed experts per token: 12 (model default 8; ranks 9..12 get linear influence decay 0.99..0.5)
#   moe: expert threshold: keep experts while p >= 0.75 x p(rank 6) (experts 3..12 per token)
#   moe: expansion active on layers 25..39 of 40
#   speed: mx.compile applied to 80 FFN sub-calls
#   speed: warmup decode — greedy <G> tok/s | sampled <S> tok/s
#   listening on http://0.0.0.0:8080  (slots: 1, ctx: 131072)
```

Apri **http://127.0.0.1:8080** per l'interfaccia chat (o usa un client OpenAI).

Velocità attesa su M-series (6-bit): decode ~50–55 tok/s col profilo di
espansione, ~65 stock. La prima risposta è più lenta (JIT Metal) — il warmup
paga quel costo all'avvio.

---

## 4. Mac 16 GB: il workflow lazy mixed-quant

Con 16 GB il checkpoint 6-bit (~28 GB) non ci sta — ma il **loader lazy
per-layer** processa un layer alla volta, quindi il picco è la dimensione
del modello MIXED (~12 GB), non della sorgente:

```bash
agrimlxo serve \
    -m mlx-community/Qwen3.5-35B-A3B-6bit \
    --mixed-from-usage --mem-target 12GB \
    --save-mixed ~/models/qwen36-mixed \
    --no-moe-expansion \
    --host 0.0.0.0 --port 7080
```

Cosa succede:
1. il 6-bit viene caricato **un layer alla volta** (lazy dal safetensors),
2. gli esperti routed di ogni layer vengono dequantizzati e requantizzati in
   4 fasce (8/6/4/2 bit) secondo il ranking di uso, dentro il target 12 GB,
3. il modello mixed viene **salvato in `~/models/qwen36-mixed/`** — i prossimi
   avvii lo caricano diretto (niente re-mix, niente picco 28 GB):

```bash
agrimlxo serve -m ~/models/qwen36-mixed --host 0.0.0.0 --port 7080
```

> **Nota:** la prima run deve comunque LEGGERE il checkpoint 28 GB dal disco
> (streaming, non in memoria). Su un Mac 16 GB il picco è il modello mixed +
> un layer sorgente (~13 GB) — ci sta.

L'heatmap USAGE (pulsante nell'header) mostra la matrice LED che ha
alimentato il ranking; **Quant map** mostra i bit assegnati (rosso 8 /
arancio 6 / giallo 4 / bianco 2). Dopo qualche conversazione, ri-esegui con
un ranking aggiornato per ribilanciare le fasce.

---

## 5. 32 / 48 / 64 GB: lo stesso flusso, fasce più alte

Il solver di `--mem-target` distribuisce le fasce 8/6/4/2 hot-first dentro
il budget — più GB = più esperti a bit più alti:

```bash
# 32 GB: base 4-bit, esperti caldi a 6–8 bit
agrimlxo serve -m <modello> --mixed-from-usage --mem-target 20GB \
    --save-mixed ~/models/qwen36-mixed --host 0.0.0.0

# 48 GB: base 6-bit (il riferimento), esperti caldi a 8-bit
agrimlxo serve -m <modello> --mixed-from-usage --mem-target 28GB \
    --save-mixed ~/models/qwen36-mixed --host 0.0.0.0

# 64 GB: tutto a 8-bit senza bisogno di mixing
agrimlxo serve -m <modello> --host 0.0.0.0
```

Monitora la memoria effettiva: la riga stderr
`mixed 4-tier: memoria modello <N> MB (target …)` — e `ps -o rss=`.

---

## 6. Se su HF non c'è il quant MLX: generalo dal bf16

```bash
agrimlxo download Qwen/Qwen3.5-35B-A3B          # sorgente bf16 (~70 GB su disco)
agrimlxo mixed-quantize ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*/ \
    --top-m 256 --hi-bits 8 --lo-bits 4 --out ~/models/qwen36-8bit
```

(Per il percorso lazy non serve: `--mixed-from-usage` lavora direttamente
sul checkpoint bf16, un layer alla volta.)

---

## 7. Tool, kache, extra

```bash
# tool (il modello può chiamare script python in ./tools/):
agrimlxo serve -m <m> --tools

# snapshot KV persistenti delle conversazioni (sopravvivono al riavvio):
agrimlxo serve -m <m> --kache --kache-budget 8GB

# disabilita il reasoning (risposte dirette):
agrimlxo serve -m <m> --no-thinking

# niente espansione (stock top-8, bit-identico):
agrimlxo serve -m <m> --no-moe-expansion
```

Extra webui: 📎 allega immagini/PDF (PDF come testo o immagini di pagina a DPI
configurabile), 🔄 rigenera una risposta, heatmap USAGE (matrice LED + quant
map per layer), pannello MOE (cambia l'espansione a runtime), cronologia
chat (sul client).

---

## 8. Problemi e soluzioni

| sintomo | soluzione |
|---|---|
| `ModuleNotFoundError: agrimlxo.models` | `git pull && pip install -e .` |
| `SwitchLinear.to_quantized() got unexpected 'mode'` | risolto in 6320d98 — `git pull` |
| `gather_qmm ... global_scale` | risolto in e0e2ddf — `git pull` |
| testo degradato dal detokenizer | risolto in 4cb7783 — `git pull` |
| swap / pressione memoria | Activity Monitor → "Memory used" + "Swap used"; abbassa `--mem-target` o usa un quant più piccolo |
| prima risposta lentissima | il warmup paga il JIT Metal all'avvio — verifica la riga `speed: warmup` |
| tutto 8-bit in Quant map su un modello 6-bit | risolto in 2d97bd0 — `git pull` |

## 9. Misurazione della qualità (protocollo paired)

Stessi prompt, `--temp 0`, UNA variabile per run: espansione OFF vs ON. Il
protocollo del paper (README → link Zenodo) misura su GPQA-Diamond; su
Qwen3.6-35B-A3B l'espansione ha dato +2.52 punti (84.34% vs 81.82%).
