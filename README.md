# Phonetic Eval

A CPU-only benchmark of speech-to-IPA (phonetic transcription) models on the
[VoxAngeles](https://github.com/) audited-IPA corpus. It runs nine model configs
over a shared audio set and scores each with **PER** (exact phone error rate) and
**PFER** (panphon articulatory-feature error rate). Lower is better.

Models: `allosaurus`, `wav2vec2phoneme`, `multipa`, `allophant`,
`zipa` (small + large, ONNX), `powsm`, `whipa` (base-cv + base-comb).

Everything runs without a GPU, without `sudo`, and without a compiler — each model
lives in its own isolated environment so their dependencies never clash.

## Results

- `PREVIOUS_RESULTS.md` — the reference numbers to verify against (160-file set).
- `summary_by_model.tsv`, `summary_by_family.tsv`, `per_file_results.tsv` — the
  latest scored run in the repo root.
- `runs/<name>/` — archived summaries for each dataset size (`160`, `balanced`, `full`).

---

## 1. Install

You need Python 3.10–3.12, plus `ffmpeg`, `git`, and `tmux`. If you can't install
those system-wide (no sudo), use the no-root micromamba route in 1a.

### 1a. Tools without root (micromamba)

```bash
# micromamba: a single static binary, no root needed
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p ~/.local/bin && mv bin/micromamba ~/.local/bin/

# ffmpeg + git + tmux into a self-contained env
~/.local/bin/micromamba create -y -p ./envs/tools -c conda-forge ffmpeg git tmux
export PATH="$PWD/envs/tools/bin:$PATH"
```

### 1b. Per-model environments

Each model gets its own venv (or micromamba env). Create only the ones you need.

```bash
# scoring (always needed): jiwer + panphon
python -m venv envs/env_score && source envs/env_score/bin/activate
pip install jiwer panphon pandas && deactivate

# allosaurus
python -m venv envs/env_allosaurus && source envs/env_allosaurus/bin/activate
pip install allosaurus torch soundfile && deactivate   # model 'uni2005' caches on first run

# HF CTC + Whisper models: wav2vec2phoneme, multipa, whipa
python -m venv envs/env_hf && source envs/env_hf/bin/activate
pip install torch torchaudio transformers librosa soundfile \
            peft panphon pandas pyyaml && deactivate

# allophant (made with uv; it ships no pip, use `uv pip`)
uv venv -p 3.11 envs/env_allophant
uv pip install --python envs/env_allophant/bin/python allophant soundfile

# zipa (ONNX path — avoids k2/icefall)
python -m venv envs/env_zipa && source envs/env_zipa/bin/activate
pip install onnxruntime torch soundfile librosa lhotse && deactivate

# powsm (ESPnet). pyworld has no wheel + no gcc here, so use micromamba:
~/.local/bin/micromamba create -y -p ./envs/env_powsm -c conda-forge \
    python=3.10 pyworld "numpy<2" compilers
envs/env_powsm/bin/pip install espnet espnet_model_zoo torch torchaudio soundfile librosa
```

### 1c. External repos + weights

```bash
mkdir -p corpora models
# ZIPA repo (for the ONNX inference helper) + checkpoints
git clone https://github.com/anyspeech/zipa corpora/zipa_repo
#  download model.onnx + tokens.txt from the anyspeech HF hubs into:
#    models/zipa_small_ns/   (anyspeech/zipa-small-crctc-ns-700k)
#    models/zipa_large_ns/   (anyspeech/zipa-large-crctc-ns-800k)

# WhIPA repo (loader class) + weights
git clone https://github.com/jshrdt/whipa corpora/whipa_repo
#  download weights into:
#    models/whipa_base_cv/      (jshrdt/whipa-base-cv)
#    models/lowhipa_base_comb/  (jshrdt/lowhipa-base-comb, LoRA adapter)
```

POWSM (`espnet/powsm`) and `openai/whisper-base` download automatically on first
run into `~/.cache/huggingface`.

### 1d. Corpus

Fetch VoxAngeles into `corpora/voxangeles/` (audited IPA lives in
`transcriptions/voxangeles_transcriptions.tsv`, audio in
`data/audited_aligned/<lang>.zip`, family maps in `lrec-coling_analyses/map/`).

---

## 2. Build the test files

The scorer reads `audio/*.wav` (16 kHz mono) and `references.tsv`
(`filename, language, family, reference_ipa` — space-separated phones). Two
builders generate both from the corpus. **Each build rewrites `audio/` and
`references.tsv` from scratch**, so only one dataset is "live" at a time.

```bash
source envs/env_score/bin/activate          # builders need panphon
export FFMPEG=envs/tools/bin/ffmpeg          # or just have ffmpeg on PATH

# balanced subset: N families x K langs x M utts (priority families first)
python scripts/build_testset.py --families 11 --langs-per-family 2 --utts 40

# OR the entire corpus (every usable utterance, ~5400 files)
python scripts/build_full.py
deactivate
```

Filenames are `<family>_<lang>_<NNN>.wav`; the family prefix is what lets the
scorer group results by family.

### Manual / custom test files

You can skip the builders and hand-make a set: drop 16 kHz mono WAVs in `audio/`
and add a matching row per file to `references.tsv`:

```
filename	language	family	reference_ipa
myclip_001.wav	Finnish	Uralic	t ä m ä
```

Phones must be space-separated. Resample anything else with
`ffmpeg -i in.wav -ar 16000 -ac 1 audio/myclip_001.wav`.

---

## 3. Run inference

### 3a. One model at a time (manual)

Run `scripts/run_inference.py` with **that model's own interpreter**; it reads
every WAV in `audio/` and writes `results/<model>/<file>.txt`.

```bash
envs/env_zipa/bin/python        scripts/run_inference.py --model zipa --variant small
envs/env_zipa/bin/python        scripts/run_inference.py --model zipa --variant large
envs/env_allosaurus/bin/python  scripts/run_inference.py --model allosaurus
envs/env_allophant/bin/python   scripts/run_inference.py --model allophant
envs/env_hf/bin/python          scripts/run_inference.py --model wav2vec2phoneme
envs/env_hf/bin/python          scripts/run_inference.py --model multipa
envs/env_powsm/bin/python       scripts/run_inference.py --model powsm
envs/env_hf/bin/python          scripts/run_inference.py --model whipa --variant base-cv
envs/env_hf/bin/python          scripts/run_inference.py --model whipa --variant base-comb
```

Notes per model:
- **zipa** — `--variant small|large`; results go to `results/zipa/` and `results/zipa_large/`.
- **whipa** — `--variant base-cv|base-comb`; results go to `results/whipa/` and `results/whipa_comb/`.
- **powsm** — slowest on CPU (~2 s/file); conditions on the ISO-639-3 code in the filename.
- **allosaurus**, **wav2vec2phoneme**, **multipa**, **allophant** — single config, no `--variant`.

### 3b. All models at once (script)

```bash
bash scripts/run_all.sh           # all 9 over the current audio/, logs to results/run_all.log
```

### 3c. Full pipeline, unattended

Builds the balanced set, runs + scores it, then does the same for the full corpus,
archiving each into `runs/<name>/`. Built to run in `tmux` so it survives a logoff:

```bash
envs/tools/bin/tmux new-session -d -s phoneval 'bash scripts/run_full_pipeline.sh'
envs/tools/bin/tmux attach -t phoneval     # watch; Ctrl-b then d to detach
cat runs/STATUS                            # progress without attaching
```

---

## 4. Score

```bash
envs/env_score/bin/python scripts/evaluate.py
column -t -s$'\t' summary_by_model.tsv
```

`evaluate.py` matches each `results/<model>/<file>.txt` to its `references.tsv`
row and writes `per_file_results.tsv`, `summary_by_model.tsv`, and
`summary_by_family.tsv`.

### Verifying against the reference numbers

To confirm a model still reproduces `PREVIOUS_RESULTS.md`: build the **160-file**
set (`--families 8 --langs-per-family 2 --utts 10`), run that model, score, and
compare its row. High PER with low PFER is expected — it's the broad-vs-narrow IPA
mismatch (references are narrow audited IPA), not a bug. See `RESUME.md` for the
full background and known gotchas.
