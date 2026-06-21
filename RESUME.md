# Phonetic Eval — Resume State

_Last updated: 2026-06-02 by Claude. Read this first when you reconnect._
_2026-06-02 session: got **Tier 3 (ZIPA + POWSM) AND WhIPA** running end-to-end on
CPU — all scored on the 48-file pilot. See "TIER 3" / "WhIPA" sections below._

## How to reconnect after an SSH drop
```bash
ssh allan@10.32.64.30           # uni VPN on
# tools (git/ffmpeg/tmux) are on PATH via ~/.bashrc now
tmux ls                          # should show 'phoneval' if the run is still going
tmux attach -t phoneval          # watch live; Ctrl-b then d to detach again
# or just read the log:
tail -f ~/phonetic_eval/results/run_tier12.log
cat ~/phonetic_eval/results/.tier12_done   # one line per finished model; "COMPLETE" at end
```
`tmux`, `git`, `ffmpeg` live in `~/phonetic_eval/envs/tools/bin` (installed with
micromamba at `~/.local/bin/micromamba`, no sudo).

## What is DONE
- **Environments** (all verified importable, CPU-only — no GPU on this box):
  - `env_allosaurus` (py3.12): allosaurus 1.0.2, torch 2.12 — model `uni2005` cached
  - `env_hf` (py3.12): torch 2.12, transformers 5.9, torchaudio, librosa, soundfile
  - `env_allophant` (py3.11, made with uv — has no `pip`, use `uv pip` or `python -m`)
  - `env_score` (py3.11): jiwer, panphon  ← scoring
  - `env_zipa` (py3.12 venv): onnxruntime 1.26, torch 2.12+cpu, lhotse, librosa,
    soundfile — ZIPA **ONNX** path (no k2/icefall needed; see Tier 3 below)
  - `env_powsm` (**micromamba** py3.10, NOT a venv): espnet + espnet_model_zoo +
    torch 2.x cpu. Made with micromamba because espnet pulls `pyworld`, which has
    no py3.12 wheel and no prebuilt wheel for this box, and **there is no gcc** to
    build it — micromamba installs conda-forge `pyworld` + compilers instead.
    Activate with `micromamba activate ~/phonetic_eval/envs/env_powsm` or just run
    `envs/env_powsm/bin/python ...` directly.
  - `envs/tools` (micromamba): git 2.54, ffmpeg 8.1.1, tmux 3.6a
- **Corpus**: VoxAngeles fetched (no git; tarball) → `corpora/voxangeles/`.
  Audited IPA = `transcriptions/voxangeles_transcriptions.tsv` (`updated` column).
- **Test set (EXPANDED 2026-06-02)**: 160 files in `audio/` + `references.tsv`.
  8 families × 2 langs × 10 utts. Families/langs: Indo-European (isl, pes),
  Atlantic-Congo (lug, cpn), Sino-Tibetan (yue, mya), Afro-Asiatic (apc, mlt),
  Austronesian (ilo, pam), Dravidian (kan, bfq), Uralic (hun, fin),
  Nakh-Daghestanian (agx, cji). Built with `scripts/build_testset.py --families 8
  --langs-per-family 2 --utts 10` (set `FFMPEG=envs/tools/bin/ffmpeg`). VoxAngeles
  can support up to 11 such families; rebuild bigger anytime (rewrites audio/ +
  references.tsv from scratch — then clear results/ and re-run). The earlier 48-
  file pilot (3 families × 2 × 8) is superseded; its numbers are in git-less
  history only (this file's prior revision).
- **Pipeline validated end-to-end**: allosaurus inference → `evaluate.py` →
  `summary_by_model.tsv`, `summary_by_family.tsv`, `per_file_results.tsv`.
  First numbers: allosaurus PER≈0.91, PFER≈0.29 (see "Known finding" below).

## RESULTS — 9 model configs, 160-file set  [updated 2026-06-02]
| model | mean PER | mean PFER | tier |
|---|---|---|---|
| powsm | 0.517 | 0.138 | 3 |
| wav2vec2phoneme | 0.530 | 0.173 | 1 |
| multipa | 0.594 | 0.144 | 1 |
| zipa (small) | 0.677 | 0.170 | 3 |
| zipa_large | 0.684 | 0.174 | 3 |
| allophant | 0.707 | 0.217 | 2 |
| whipa_comb (base-comb) | 0.831 | 0.225 | 3 |
| whipa (base-cv) | 0.837 | 0.277 | 3 |
| allosaurus | 0.921 | 0.307 | 1 |

**POWSM is best on BOTH PER and PFER** on this set. Indo-European PER stays high
for everyone (narrow Icelandic refs — see "Known finding"); POWSM's best family is
Uralic (fin/hun, PER 0.29). Findings worth noting vs the 48-file pilot:
- **ZIPA small ≈ large** at scale (0.677 vs 0.684, per-family mixed) — the pilot's
  "large clearly better" did NOT hold. Don't over-claim a size effect.
- **WhIPA base-cv is dragged up by a Hungarian blow-up** (hun PER=1.83 — runaway
  hallucination longer than the reference, which slipped UNDER the 20 phones/sec
  fallback threshold so the fallback never fired). `whipa_comb` is more stable
  (hun 0.98) and edges ahead overall. WhIPA's best langs (ilo, kan, apc) are NOT
  its training langs — at whisper-base size, broad/narrow mismatch dominates any
  in-domain advantage. If pushing WhIPA further, lower `max_phones_per_sec_rate`
  in `run_whipa()` (currently 20) to make the fallback catch these, and/or use the
  whisper-large-v2 variants.

Full breakdown in `summary_by_model.tsv`, `summary_by_family.tsv`,
`per_file_results.tsv`. Re-run everything over the current audio/ with
`bash scripts/run_all.sh` (logs to `results/run_all.log`; runs all 9 configs in
their envs, ~15 min on CPU). To re-run one model: `source envs/<env>/bin/activate
&& python scripts/run_inference.py --model <name> [--variant ...]`. Re-score with:
```bash
cd ~/phonetic_eval && source envs/env_score/bin/activate
python scripts/evaluate.py && deactivate
column -t -s$'\t' summary_by_model.tsv
```

### Bugs found & fixed during the pilot (all patched in scripts/)
- **torchaudio.load needs torchcodec** (not installed) in torch 2.x → rewrote the
  audio loader to use `soundfile` (audio is already 16k mono). Added `soundfile`
  to env_allophant.
- **wav2vec2phoneme tokenizer needs phonemizer/espeak** just to load → load the
  processor with `do_phonemize=False` (we only CTC-decode, never phonemize), so no
  espeak dependency. (`phonemizer` is pip-installed in env_hf but unused now.)
- **allophant decode** → implemented the real path from the HF model card:
  `predictions.feature_decoders(indexer, feature_names=["phoneme"])` then
  `inventory_indexer.feature_values("phoneme", tokens-1)`, decoding against a fixed
  broad multilingual inventory (`ALLOPHANT_INVENTORY_LANGS` in run_inference.py) for
  the language-independent setting. NOTE: allophant's own CLI is broken in this env
  (numcodecs/zarr clash) — we use the Estimator API directly, so it doesn't matter.

## TIER 3 — DONE (ZIPA + POWSM), CPU-only  [2026-06-02]
The guide warned Tier 3 was hard (k2/CUDA, heavy ESPnet). Both turned out runnable
on this CPU box via paths that sidestep the hard deps:

**ZIPA** — use the repo's **ONNX** inference path, NOT the torch path. The torch
path (`zipa_ctc_inference.py`) imports `icefall`/`k2`; the ONNX path needs only
`onnxruntime soundfile librosa lhotse torch`. Repo cloned to `corpora/zipa_repo`.
ONNX checkpoints (`model.onnx` + `tokens.txt`) downloaded from the anyspeech HF
hubs into `models/zipa_small_ns/` (anyspeech/zipa-small-crctc-ns-700k) and
`models/zipa_large_ns/` (anyspeech/zipa-large-crctc-ns-800k). Vocab = 127 per-char
IPA symbols, so CTC-greedy output is already space-separated phones; we drop the
sentencepiece word-boundary marker `▁` (U+2581). ZIPA emits modifiers/diacritics
(ʰ ː ˞ ̪) as separate tokens — kept as-is (PFER re-segments them; PER does not).
```bash
source ~/phonetic_eval/envs/env_zipa/bin/activate
python scripts/run_inference.py --model zipa --variant large   # or --variant small
deactivate    # ~4s (small) / ~8s (large) for all 48 files
```

**POWSM** — ESPnet S2T model, task `<pr>`. Runs on CPU via
`espnet2.bin.s2t_inference.Speech2Text.from_pretrained("espnet/powsm")`. Install
gotcha: espnet needs `pyworld`, no py3.12/prebuilt wheel + no gcc on this box, so
`env_powsm` is a **micromamba** env (py3.10, conda-forge pyworld+compilers) — see
env list above. Output is slash-wrapped phones (`/p//e//iː/`); we split on `/` to
recover space-separated phones (NOT the model card's `replace("/","")` which
concatenates them). POWSM conditions on a per-utterance language symbol: we feed
the file's ISO-639-3 code (filename middle field) when POWSM knows it, else its
built-in `<unk>`. **Caveat:** several test langs are NOT in POWSM's 92-lang
inventory (e.g. pes/lug/cpn and most VoxAngeles ISO codes) and fall back to
`<unk>` — but this did NOT obviously hurt them (on the pilot, lug+unk was POWSM's
best language; isl with a real symbol was among the worst — the narrow-IPA effect
dominates). Note the asymmetry when writing up.
```bash
envs/env_powsm/bin/python scripts/run_inference.py --model powsm   # ~4-5 min for 160 files
```
Both feed into the same scorer; re-run `evaluate.py` (env_score) and they appear
as `powsm`, `zipa`, `zipa_large` in the summaries.

## WhIPA — DONE (Whisper fine-tuned for IPA), CPU  [2026-06-02]
WhIPA is NOT a drop-in HF pipeline; it loads through the jshrdt/whipa repo's
`WHIPA` class (`corpora/whipa_repo/code/deploy.py`). The fine-tuned weights ARE
public after all (the collection `jshrdt/lowhipa-models`; the old 401 was for a
different/older path). Wired into `run_inference.py::run_whipa(variant)`:
- `base-cv`  → `jshrdt/whipa-base-cv`  (full fine-tune, no LoRA) → `results/whipa/`
- `base-comb`→ `jshrdt/lowhipa-base-comb` (LoRA adapter, needs peft) → `results/whipa_comb/`
Weights in `models/whipa_base_cv/` and `models/lowhipa_base_comb/`. All variants
use `openai/whisper-base` as the base (downloaded by transformers on first run).
```bash
source ~/phonetic_eval/envs/env_hf/bin/activate
python scripts/run_inference.py --model whipa --variant base-cv     # ~55s / 48 files
python scripts/run_inference.py --model whipa --variant base-comb   # ~110s (LoRA)
```
Deps added to env_hf: `peft panphon pandas pyyaml` (deploy.py imports peft;
`scripts.metrics.retokenize_ipa` needs panphon/pandas/yaml).

Implementation notes / gotchas:
- The repo's `transcribe_ipa()` is BUGGY (references an undefined global `whipa`
  and uses `torch` without importing it). We reuse WHIPA only for *loading* (it
  correctly installs the custom `<|ip|>` "IPA language" token + tokenizer) and
  reimplement the decode + phone-rate fallback ourselves, fixed, decoding with
  `skip_special_tokens=True`. Fallback ladder = beam backoff → repetition_penalty
  → exponential_decay_length_penalty → truncate; triggers only if phones/sec > 20
  (it does NOT fire on our short clips). We build `input_features` straight from
  `whipa.processor`, so the repo's `datasets`/`prep_dataset` path is not needed.
- **Why WhIPA scores low here:** both variants are whisper-**base** (74M, chosen
  for CPU speed) and NEITHER was trained on our test languages — `base-cv` saw
  only Common Voice {ja,pl,mt,hu,fi,el,ta}; `base-comb` added Arabic+Mandarin but
  still none of isl/pes/lug/cpn/yue/mya. So these are honest zero-shot numbers.
  The paper's strong results use **whisper-large-v2** (`whipa-large-cv`,
  `lowhipa-large-comb`); those run with this same code (base auto-inferred as
  whisper-large-v2) but are ~1.5B params → very slow on CPU. Add a `large-*`
  entry to `WHIPA_VARIANTS` if a GPU appears or you can spare the CPU time.

## Known finding (not a bug)
High PER with low PFER = broad-vs-narrow mismatch. References are NARROW audited
IPA (length marks `ː`, devoicing `b̥`, dentals `t̪`); models output BROAD phones.
Exact-match PER punishes `iː`≠`i`, `b̥`≠`b`; feature-based PFER stays low because
the phones are articulatorily close. Report this; don't silently normalize it
away (see guide §9). If you WANT a diacritic-folded variant, add it as a second
metric in `evaluate.py` rather than replacing PER.

## What is PENDING / decisions for you
- **Expand the test set** — ✅ DONE 2026-06-02 (now 8 families × 2 langs × 10 utts
  = 160 files; see "Test set (EXPANDED)" above). Could go to 11 families or more
  utts/lang (median 49 available): rebuild with `build_testset.py`, clear
  `results/*/`, then `bash scripts/run_all.sh` and re-score.
- **WhIPA** — ✅ DONE 2026-06-02 (base-cv + base-comb, both whisper-base). See the
  "WhIPA — DONE" section above. Follow-up: run the whisper-large-v2 variants for
  the paper's headline numbers (needs GPU or patience — same code, just add a
  `large-*` entry to WHIPA_VARIANTS pointing at `jshrdt/whipa-large-cv` /
  `lowhipa-large-comb`).
- **Allophant decode** — `run_inference.py::run_allophant()` has best-effort decode
  logic with version-dependent fallbacks. Check its output in
  `results/allophant/*.txt`; if it's empty/garbage, print `model_outputs` once and
  fix the decode call per the installed allophant version.
- **Tier 3 (ZIPA, POWSM)** — ✅ DONE 2026-06-02, both run on CPU & scored. See the
  "TIER 3 — DONE" section above. Possible follow-ups: run ZIPA's no-diacritics
  checkpoints (anyspeech/*-no-diacritics-*) for a fairer broad-vs-narrow PER, or
  the int8 ONNX for speed; nothing blocking.
- **Other corpora** — DoReCo (license-gated manual download + X-SAMPA→IPA via
  jhasegaw/phonecodes), THCHS-30 (needs Taubert-2023 IPA layer), ASC (Buckwalter→IPA
  via WhIPA repo). All blocked on manual/licensing steps — see guide §5.

## Key paths
- scripts: `~/phonetic_eval/scripts/{run_inference.py, evaluate.py, build_testset.py, run_tier12.sh}`
  (run_inference.py now supports models: allosaurus, wav2vec2phoneme, multipa,
  whipa [--variant base-cv|base-comb], allophant, zipa [--variant small|large], powsm)
- models (Tier 3 weights): `~/phonetic_eval/models/{zipa_small_ns,zipa_large_ns}/`
  (model.onnx + tokens.txt), `models/{whipa_base_cv,lowhipa_base_comb}/` (WhIPA).
  POWSM + base Whisper weights are cached under `~/.cache/huggingface`.
- repos: `~/phonetic_eval/corpora/zipa_repo/` (ZIPA; ONNX inference in `inference/`),
  `corpora/whipa_repo/` (WhIPA; loader class in `code/deploy.py`)
- results: `~/phonetic_eval/results/<model>/<file>.txt` + `*.tsv` summaries in project root
- papers: `~/phonetic_eval/papers/` (Allosaurus, W2V2Phoneme, Allophant, MultIPA, ZIPA, POWSM, STIPA, WhisperPPT, XIPA, gao21)
