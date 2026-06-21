# Phonetic Transcription Evaluation — Full Setup & Execution Guide

This guide walks through evaluating universal speech-to-IPA models on a remote
Linux server. It is written to be self-contained: you can follow it top to
bottom, and you can paste any section into Claude CLI when you need help with a
specific step.

**Environment:** any Linux machine with Python 3.10–3.12 (CPU is fine).
**Goal:** run several phonetic transcription models on shared test audio, then
score them with Phone Error Rate (PER) and Phonological Feature Error Rate (PFER).
**Test corpora chosen:** VoxAngeles, DoReCo, THCHS-30, ASC.

---

## Table of Contents

0. Important concepts (read once)
1. Get onto your machine
2. Create the project folder structure
3. Python environments (one per model family — why and how)
4. Install the models
5. Install / obtain the evaluation corpora
6. Build the test set (references.tsv + audio)
7. Run inference (per model)
8. Score with PER and PFER (panphon + jiwer)
9. Known output-format issues and fixes ("fine-tuning")
10. (Stretch goal) Training your own model — notes for later

---

## 0. Important concepts (read once)

- **You work in a Linux bash shell.** All commands below are Linux shell commands
  (`ls`, `cd`, `pwd`, `mkdir`). How you reach that shell (local terminal, SSH to a
  server, etc.) is up to your setup.
- **One model per environment.** These projects pin conflicting versions of
  PyTorch, transformers, and (for ZIPA) the k2 library. Installing them all in
  one environment will break things. We create a separate virtual environment
  per model family. Inference for each model is run in its own environment;
  outputs are plain text files, so the scoring step (separate environment) can
  read all of them regardless of which environment produced them.
- **The filename is the join key.** Every audio file `X.wav` has exactly one
  reference row keyed `X`, and each model writes its output to
  `results/<model>/X.txt`. Keep this consistent and everything downstream works.
- **tmux keeps jobs alive.** SSH sessions die if your laptop sleeps. Run long
  jobs inside `tmux` so they survive disconnects (see section 1).

---

## 1. Get onto your machine

This benchmark runs on any Linux box with Python 3.10–3.12. How you connect is
specific to your environment (a local terminal, SSH to a remote server, a
container, …) and is out of scope here — the rest of this guide assumes you have
a shell on that machine and start in your home directory.

Run long jobs inside `tmux` so they survive a dropped connection:

```bash
tmux new -s phoneval          # later, reattach with:  tmux attach -t phoneval
```

---

## 2. Create the project folder structure

Inside the SSH session:

```bash
cd ~
mkdir -p phonetic_eval/audio
mkdir -p phonetic_eval/results
mkdir -p phonetic_eval/corpora        # raw downloaded corpora live here
mkdir -p phonetic_eval/envs           # virtual environments live here
mkdir -p phonetic_eval/scripts        # run_inference.py, evaluate.py
cd phonetic_eval
```

Final intended layout:

```
phonetic_eval/
├── audio/                  # the test WAVs (16kHz mono), shared by all models
├── references.tsv          # filename, language, family, reference_ipa
├── corpora/                # raw downloaded corpora (voxangeles, doreco, ...)
├── envs/                   # env_allosaurus, env_hf, env_allophant, ...
├── scripts/                # run_inference.py, evaluate.py
└── results/                # results/<model>/<filename>.txt
```

---

## 3. Python environments (one per model family)

Check what Python is available first:

```bash
python3 --version          # need >=3.10 for Allophant; >=3.8 for the others
which python3
```

We create one environment per model family. Each is created the same way:

```bash
# template — do NOT run as-is, see per-model commands in section 4
python3 -m venv ~/phonetic_eval/envs/ENVNAME
source ~/phonetic_eval/envs/ENVNAME/bin/activate
pip install --upgrade pip wheel
# ... install that model's dependencies ...
deactivate
```

Planned environments:

| Environment | Models it serves | Reason for separation |
|---|---|---|
| `env_allosaurus` | Allosaurus | pins older torch; isolate to be safe |
| `env_hf` | Wav2Vec2Phoneme, MultIPA, WhIPA | all use recent `transformers` + `torch`; compatible together |
| `env_allophant` | Allophant | needs Python >=3.10 and its own dependency set |
| `env_zipa` | ZIPA | needs k2/icefall pinned to exact torch+CUDA; must be isolated |
| `env_powsm` | POWSM | needs espnet (heavy); isolate |

Activate an environment with `source ~/phonetic_eval/envs/<name>/bin/activate`
and leave it with `deactivate`. Always confirm which environment is active
(its name appears in the prompt) before installing or running.

---

## 4. Install the models

Models are tiered by install difficulty. Get Tier 1 working end-to-end first;
only attempt Tier 3 if you have time. **Always read each repo's README on the
server and cross-check its dependency list before installing** — versions drift.

### Tier 1 — straightforward

**Allosaurus** (own environment):
```bash
python3 -m venv ~/phonetic_eval/envs/env_allosaurus
source ~/phonetic_eval/envs/env_allosaurus/bin/activate
pip install --upgrade pip wheel
pip install allosaurus
# smoke test (downloads the default model on first run):
python -m allosaurus.run -i ~/phonetic_eval/audio/SOME_FILE.wav
deactivate
```

**Wav2Vec2Phoneme + MultIPA + WhIPA** (shared HuggingFace environment):
```bash
python3 -m venv ~/phonetic_eval/envs/env_hf
source ~/phonetic_eval/envs/env_hf/bin/activate
pip install --upgrade pip wheel
pip install torch torchaudio transformers soundfile librosa
deactivate
```
Checkpoints (loaded by the script, not installed separately):
- Wav2Vec2Phoneme: `facebook/wav2vec2-xlsr-53-espeak-cv-ft`
- MultIPA: confirm exact name in https://github.com/ctaguchi/multipa README
  (e.g. `ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns`)
- WhIPA: confirm exact name in https://github.com/jshrdt/whipa README

### Tier 2 — moderate

**Allophant** (own environment, Python >=3.10):
```bash
python3 -m venv ~/phonetic_eval/envs/env_allophant
source ~/phonetic_eval/envs/env_allophant/bin/activate
pip install --upgrade pip wheel
pip install allophant
deactivate
```
Allophant's inference API is more involved than the others (it needs a phoneme
inventory and a Batch object); the correct usage is implemented in
`run_inference.py` (section 7) based on the official README.

### Tier 3 — hard (attempt only if Tier 1 & 2 are done)

**ZIPA** — needs the k2 library, which must match the server's exact PyTorch and
CUDA versions. Budget a full day and do not let it block the rest.
```bash
# Inspect first:
git clone https://github.com/lingjzhu/zipa ~/phonetic_eval/corpora/zipa_repo
# Read its README for the exact k2/icefall/torch versions required, then build
# env_zipa accordingly. If k2 will not install, cite ZIPA's published numbers
# in the survey instead of running it yourself.
```

**POWSM** — needs ESPnet.
```bash
git clone https://github.com/espnet/espnet ~/phonetic_eval/corpora/espnet_repo
# Follow ESPnet's install guide; the model is espnet/powsm on HuggingFace.
# Heavy install; attempt last.
```

---

## 5. Install / obtain the evaluation corpora

You chose four. Two have ready-made trustworthy IPA; two require sourcing or
building the IPA layer. This is important to know up front.

### VoxAngeles — ready IPA, easiest
```bash
cd ~/phonetic_eval/corpora
git clone https://github.com/pacscilab/voxangeles
# Inspect structure:
ls voxangeles
# Look for the audited transcriptions and per-language audio. The audited IPA
# is already canonical IPA — no conversion needed.
```

### DoReCo — naturalistic, IPA via X-SAMPA conversion
DoReCo is downloaded per language from https://doreco.huma-num.fr/ (you accept a
licence and download bundles). There is no simple one-line CLI. Recommended:
download the 3-4 language bundles you want, then place them under
`corpora/doreco/` (if you downloaded them elsewhere, copy them across with `scp`).
DoReCo transcriptions are in **X-SAMPA** and need conversion to IPA. Use the
Python library `jhasegaw/phonecodes` (academically citable):
```bash
source ~/phonetic_eval/envs/env_hf/bin/activate    # reuse any env
pip install panphon                                # also needed for scoring
git clone https://github.com/jhasegaw/phonecodes ~/phonetic_eval/corpora/phonecodes
# Use phonecodes to convert X-SAMPA -> IPA, then spot-check 10-15 lines per
# language by eye against an X-SAMPA/IPA chart (your phonetics training matters
# here — watch the length mark becomes U+02D0 'ː', not a colon).
deactivate
```

### THCHS-30 — audio free, IPA layer must be sourced
```bash
cd ~/phonetic_eval/corpora
wget https://openslr.org/resources/18/data_thchs30.tgz
tar -xzf data_thchs30.tgz
```
**Caveat:** base THCHS-30 ships with Chinese characters and pinyin, NOT IPA. The
trustworthy phone-level IPA used by STIPA comes from Taubert (2023), a separate
release you must locate. Without it, you would fall back to G2P, which defeats
the purpose of using THCHS as a gold set. Resolve this before committing to THCHS.

### ASC (Arabic Speech Corpus) — audio free, IPA layer is rule-based
The corpus is downloadable from http://en.arabicspeechcorpus.com/ . The IPA used
by STIPA was produced by their own Buckwalter→IPA transliteration module, found
in the WhIPA repo (https://github.com/jshrdt/whipa). To reproduce their IPA layer
you would run that module on the corpus's Buckwalter transcripts. Check the WhIPA
repo for the script before committing to ASC.

**Practical recommendation:** start with VoxAngeles end-to-end (cleanest). Add
DoReCo second (naturalistic, your strongest research-question angle). Treat
THCHS-30 and ASC as additions only after the IPA-layer sourcing above is solved.

---

## 6. Build the test set (references.tsv + audio)

All models read the same WAVs from `audio/`, resampled to 16kHz mono. Do the
resampling once (install ffmpeg if missing: `sudo apt install ffmpeg`, or ask
the professor if you lack sudo — `pip install ffmpeg-python` is a fallback).

Resample every source file into `audio/` with a family-prefixed name:
```bash
# example: convert one file
ffmpeg -i corpora/voxangeles/some_lang/word001.wav -ar 16000 -ac 1 \
  audio/indoeuropean_pol_001.wav
```

Naming convention: `family_lang_NNN.wav` (e.g. `turkic_tat_003.wav`). The family
prefix lets you group results by language family later, which is one of your
research questions.

`references.tsv` — one row per audio file, tab-separated, IPA space-separated by
phone:
```
filename	language	family	reference_ipa
indoeuropean_pol_001.wav	Polish	Indo-European	ɛ s t ɛ ʂ
turkic_tat_003.wav	Tatar	Turkic	k ɪ t a p
```

Build this file locally (it is easier to edit on your laptop) and `scp` it up,
or edit on the server with `nano references.tsv`. Scope target: 3-4 families,
2 languages each, ~10-15 utterances per language (60-120 files total).

---

## 7. Run inference (per model)

Use the `run_inference.py` script (provided as a separate file alongside this
guide). It is run **once per model, inside that model's environment**. Because
each model's import happens lazily inside its own function, only the selected
model's library needs to be present in the active environment.

```bash
cd ~/phonetic_eval/scripts

# Allosaurus
source ~/phonetic_eval/envs/env_allosaurus/bin/activate
python run_inference.py --model allosaurus
deactivate

# HuggingFace models (in env_hf)
source ~/phonetic_eval/envs/env_hf/bin/activate
python run_inference.py --model wav2vec2phoneme
python run_inference.py --model multipa
python run_inference.py --model whipa
deactivate

# Allophant
source ~/phonetic_eval/envs/env_allophant/bin/activate
python run_inference.py --model allophant
deactivate
```

Each run fills `results/<model>/<filename>.txt` with one IPA string per audio
file. Run inside tmux; the first run of each model downloads its weights.

---

## 8. Score with PER and PFER (panphon + jiwer)

Scoring is a single environment that only needs `jiwer` and `panphon` — it reads
the plain-text outputs from every model regardless of which environment produced
them.

```bash
python3 -m venv ~/phonetic_eval/envs/env_score
source ~/phonetic_eval/envs/env_score/bin/activate
pip install --upgrade pip wheel
pip install jiwer panphon
cd ~/phonetic_eval
python scripts/evaluate.py
deactivate
```

This produces:
- `per_file_results.tsv` — every file × model, with PER and PFER
- `summary_by_model.tsv` — mean PER and PFER per model
- `summary_by_family.tsv` — mean per model per language family (your RQ on
  whether performance varies by family)

---

## 9. Known output-format issues and fixes ("fine-tuning")

panphon expects canonical Unicode IPA segments. Model outputs rarely come out
clean. Expect to handle these (the scoring script does basic cleaning; extend it
as you discover issues):

- **Word/space delimiters.** Allosaurus separates phones with spaces; some HF
  models emit word boundaries or `|`. The script strips spaces before panphon
  (which segments internally) and keeps spaces for jiwer (which needs them as
  phone delimiters). If a model emits no spaces at all, jiwer's PER will be wrong
  until you segment the output into phones (use panphon's `ipa_segs` to insert
  spaces).
- **Special tokens.** Whisper-based models (WhIPA) may emit language tags,
  `<|...|>` tokens, or punctuation. Strip these before scoring.
- **Unicode normalization.** The same phone can have different code points
  (e.g. 'g' U+0067 vs 'ɡ' U+0261; length ':' vs 'ː' U+02D0). Normalize both
  reference and hypothesis with `unicodedata.normalize("NFD", s)` and map common
  look-alikes, or panphon will count identical-looking phones as errors.
- **Broad vs narrow transcription.** Most models output broad transcription; if
  your reference is narrow (lots of diacritics), PFER will be inflated for
  reasons that are not the model's fault. Note this in your analysis rather than
  "fixing" it — it is itself a finding.
- **Empty outputs.** A model may return an empty string on hard audio. The
  script warns on missing files; decide whether empty = max error or excluded.

Validate your whole pipeline by reproducing a published number: run a model on a
corpus it was evaluated on in its paper and check your PER/PFER is in the same
ballpark. If it is wildly off, the problem is almost always tokenization or
normalization here, not the model.

---

## 10. (Stretch goal — for later) Training your own model

Not for now. If you finish the evaluation and have time, the MultIPA-style
approach is feasible: fine-tune `wav2vec2-large-xlsr-53` with a CTC head on a
small, high-quality dataset, using an 80/20 train/eval split.

Key cautions to design in from the start:
- **No leakage.** If you train and evaluate on the same corpus, split by
  *speaker* and by *utterance* so no utterance (or ideally speaker) appears in
  both halves. Otherwise your eval numbers are meaningless.
- **The eval sets are small.** VoxAngeles is isolated words; DoReCo subsets are
  small. An 80/20 split of a small set gives a tiny, high-variance test set.
  MultIPA worked with ~1k examples per language, so this is plausible but expect
  noisy results.
- **Keep it honest as a contribution.** Framing it as "can a small, clean,
  human-transcribed training set rival models trained on 17k hours of noisy G2P
  data?" connects directly to your survey's central argument and would be a
  genuinely interesting result either way.
- This is a separate environment (reuse `env_hf` plus `pip install datasets
  evaluate accelerate`). Ask Claude CLI for a training script when you reach this
  stage — design the split logic first, before any training code.
