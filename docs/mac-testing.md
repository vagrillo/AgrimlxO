# Test su Mac — checklist manuale

Guida operativa per testare AgrimlxO su Apple Silicon (i test di
esecuzione/compilazione sono manuali, per progetto). I 61 test unitari
passano anche su Linux/CPU; ciò che va verificato **solo su Mac** è:
Metal/GatedDeltaNet, i kernel nativi, i tempi reali e la qualità
dell'espansione sul modello 35B.

## 0. Requisiti

```bash
sw_vers                      # macOS 14+ consigliato
xcode-select --install       # serve per i kernel Metal (punto 5)
python3 --version            # 3.11–3.13
```

## 1. Installazione

```bash
git clone https://github.com/vagrillo/AgrimlxO && cd AgrimlxO
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# suite unitaria (61 test — deve finire verde anche sul Mac)
pytest tests/ -q
```

## 2. Modello

Serve un checkpoint in formato MLX. Casistiche:

| Mac (memoria unificata) | cosa usare |
|---|---|
| ≥ 96 GB | bf16 nativo: `agrimlxo download Qwen/Qwen3.6-35B-A3B` (scarica i safetensors HF; il loader li legge direttamente) |
| 48–96 GB | quant 8-bit MLX (es. `mlx-community/Qwen3.6-35B-A3B-8bit` se presente) |
| 24–48 GB | quant 4-bit MLX |
| 16–24 GB | quant 2–3-bit MLX (cerca su HF: `mlx-community` / `ISTA-DASLab`); atteso perimento ridotto — è il caso "low-bit" del paper |

```bash
agrimlxo download <hf-repo-id>   # scarica nel cache HF
agrimlxo list                    # mostra i modelli locali + geometria MoE
agrimlxo serve --suggest-quant   # suggerisce il quant più grande che entra nella memoria
```

Se su HF non c'è ancora un quant MLX del 35B, si può generare un 8-bit
uniforme senza mlx-lm (converte tutto il routed branch):

```bash
agrimlxo download Qwen/Qwen3.6-35B-A3B
agrimlxo mixed-quantize ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/ \
    --top-m 256 --hi-bits 8 --lo-bits 4 --out ~/models/qwen36-35b-8bit
```

## 3. Servire e verificare l'espansione (cuore del progetto)

```bash
# profilo default AgrillaMoE iniettato automaticamente per Qwen3.6-35B-A3B
# (esperti 12, soglia 0.75, layer 25-39); in stderr deve comparire:
#   moe: default expansion profile injected (Qwen3.5/3.6-35B-A3B): ...
#   moe: routed experts per token: 12 (model default 8; ranks 9..12 get linear influence decay 0.99..0.5)
#   moe: expert threshold: keep experts while p >= 0.75 x p(rank 6) (experts 3..12 per token)
#   moe: expansion active on layers 25..39 of 40
agrimlxo serve -m <modello>
```

Verifiche mirate, un parametro alla volta (una variabile, greedy):

```bash
# stock (bit-identico): nessuna riga "moe:" nel banner
agrimlxo serve -m <m> --no-moe-expansion

# N fisso (senza soglia): la stats deve dire esattamente 20.0
MOE_EXPERT_STATS_EVERY=64 agrimlxo serve -m <m> --moe-experts 20

# soglia adattiva: media attesa ~15–16 con N=20 T=0.8
agrimlxo serve -m <m> --moe-experts 20 --moe-expert-threshold 0.8 --moe-stats-every 64

# floor: T=10 → esattamente N/4 = 5 esperti/token
agrimlxo serve -m <m> --moe-experts 20 --moe-expert-threshold 10 --moe-stats-every 64

# potatura sotto il top-K nativo
agrimlxo serve -m <m> --moe-experts 6 --moe-expert-threshold 2

# decay: confronta greedy a parità di prompt con decay-end 0.5 vs 0.3 vs --moe-no-expert-decay
agrimlxo serve -m <m> --moe-experts 20 --moe-expert-decay-end 0.3
agrimlxo serve -m <m> --moe-experts 20 --moe-no-expert-decay

# layer-scoped: la stats deve mostrare solo L25..L39
agrimlxo serve -m <m> --moe-experts 20 --moe-expert-layer-start 25 --moe-expert-layer-end 39 --moe-stats-every 64

# renorm: auto (=stock per Qwen) vs always vs never (greedy deve cambiare con never)
agrimlxo serve -m <m> --moe-experts 20 --moe-expert-renorm never

# reduce (l'inverso): solo gli esperti con peso >= X·peso(top-1)
agrimlxo serve -m <m> --moe-reduce 0.5 --moe-reduce-layers 25-39

# errori attesi (exit non-zero, messaggio chiaro):
agrimlxo serve -m <m> --moe-experts 999          # N fuori range
agrimlxo serve -m <m> --moe-expert-threshold 0.8 # soglia senza N
agrimlxo serve -m <m> --moe-experts 20 --moe-reduce 0.5   # mutuamente esclusivi

# alias ds4 equivalenti:
agrimlxo serve -m <m> --q35-experts 20 --q35-expert-threshold 0.8
```

La riga health-signal per token processati (prompt+generati):

```
moe: experts/token avg: L25:16.1 L26:15.4 ... L39:14.9 | mean 15.5
```

Con `N=20, T=0.8` aspettarsi media ~15–16; con `T=0` esattamente 20.0;
con `T=10` esattamente 5.0. `/props` e `/metrics` espongono lo stato
dell'espansione anche via HTTP; la webui (http://127.0.0.1:8080) mostra il
badge "MoE expansion".

## 4. API e webui

```bash
curl -s localhost:8080/health
curl -s localhost:8080/props | python3 -m json.tool   # guarda moe_expansion
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Cosè un MoE?"}],"max_tokens":128,"stream":false}'
# streaming SSE + reasoning_content (i <think> del Qwen3.x separati dal contenuto):
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Risolvi: 17*23"}],"stream":true,"stream_options":{"include_usage":true}}' | head -40
```

Apri http://127.0.0.1:8080 : chat streaming, markdown, pannello ragionamento
collassabile, badge espansione, tok/s. Secondo turno della stessa chat deve
riusare la cache del prefisso (si vede dal prompt_tokens ridotto in usage
... il riuso è interno; la verifica forte è la coerenza delle risposte).

Reasoning budget (wrap-up forzato quando il `<think>` sfora):

```bash
agrimlxo serve -m <m> --reasoning-budget 256
# il <think> si deve chiudere entro ~budget anche su domande aperte
```

## 5. Kernel nativi (Metal, opt-in)

```bash
xcrun -f metal && xcrun -f metallib        # entrambi devono stampare un path
AGRIMLXO_WITH_KERNELS=1 pip install ./native
agrimlxo serve -m <m> --fast-moe-prefill
# stderr atteso: "kernels: fused MoE weighted-sum enabled for large prefill batches"
```

Verifica funzionale: prompt lungo (≈4k+ token, es. incolla un documento e
chiedi un riassunto) con e senza `--fast-moe-prefill` — output identico a
meno del riordino fp, prefill più rapido. Se il kernel non è in grado su
una forma, il server torna alle ops stock con un warning e continua.

Il kernel accetta qualsiasi budget dell'espansione: prova prefill lungo con
`--moe-experts 20` attivo (i budget 12/16/20/24/32 hanno kernel tiled dedicati).

## 6. Quant mista per-esperto

```bash
# ranking dagli usage reali (dopo una sessione con --moe-expert-log):
agrimlxo serve -m <m> --moe-expert-log run.jsonl
# (mo_predict.py del repo AgrillaMoE/llama.cpp fork legge run.jsonl ed emette i ranking)

agrimlxo mixed-quantize <modello> --order expert-order.txt \
    --top-m 64 --hi-bits 8 --lo-bits 4 --out ~/models/qwen36-mixed

agrimlxo serve -m ~/models/qwen36-mixed --moe-experts 20 --moe-expert-threshold 0.8
```

Controlli: la memoria del processo scende (il tail è a 4-bit) rispetto
all'uniforme; output greedy coerente con l'uniforme 8-bit sui prompt semplici;
con `--moe-experts` attivo l'espansione attraversa i gruppi senza errori.
Memoria prima/dopo: `ps -o rss= -p $(pgrep -f agrimlxo)`.

## 7. Qualità (protocollo paired del paper)

Per misurare l'effetto (non solo che funzioni): stessi prompt, greedy
(`--temp 0`), una variabile per run — es. GPQA-Diamond o il tuo benchmark:
stock vs `--moe-experts 20 --moe-expert-threshold 0.8` (profilo default).
Il protocollo completo è nel paper citato nel README
(<https://zenodo.org/records/22255483>) e in `docs/moe-expansion.md`.

## Problemi noti / cosa riportare

- segfault: `lldb -- agrimlxo serve -m <m> ...` + backtrace + righe banner `moe:`
- kernel: log esatto di `agrimlxo serve -m <m> --fast-moe-prefill`
- performance: tok/s prefill/decode da webui o usage, con/without expansion
  (attesa ~N/K sul routed branch; è un cambio di routing, non un speedup)
