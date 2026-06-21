# Phonetic Eval

A reproducible, **CPU-only** benchmark of speech-to-IPA (phonetic transcription)
models. It runs nine model configurations over a shared set of audio clips and
scores each against audited IPA references using two metrics — **PER** (phone
error rate) and **PFER** (phonological feature error rate). Lower is better.

The project is built to run on a modest Linux box with **no GPU, no `sudo`, and
no compiler**: every model lives in its own isolated environment so their
conflicting dependencies never meet.

## Table of Contents

1. [What this does](#1-what-this-does)
2. [The paper this is based on](#2-the-paper-this-is-based-on)
3. [Models evaluated](#3-models-evaluated)
4. [Results: our numbers vs. the paper](#4-results-our-numbers-vs-the-paper)
5. [How the evaluation works](#5-how-the-evaluation-works)
6. [Repository layout](#6-repository-layout)
7. [Setup — automatic vs. manual](#7-setup--automatic-vs-manual)
8. [Building the test set](#8-building-the-test-set)
9. [Running inference](#9-running-inference)
10. [Scoring (PER & PFER)](#10-scoring-per--pfer)
11. [Reproducing & verifying the numbers](#11-reproducing--verifying-the-numbers)
12. [Notes, gotchas & license](#12-notes-gotchas--license)

---

## 1. What this does

Given a folder of short speech clips and their reference IPA, the pipeline:

1. **Builds a test set** — extracts audio + audited IPA from a corpus into
   `audio/*.wav` (16 kHz mono) and `references.tsv`.
2. **Runs inference** — each model transcribes every clip into IPA, written to
   `results/<model>/<clip>.txt`.
3. **Scores** — compares every hypothesis to its reference and produces
   `summary_by_model.tsv`, `summary_by_family.tsv`, and `per_file_results.tsv`.

You can run the whole thing with **one script**, or do every step **by hand** to
understand or debug it. Both paths are documented below.

[↑ Back to top](#table-of-contents)

---

## 2. The paper this is based on

This benchmark replicates the evaluation design of:

> **Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support
> Language Documentation at Scale** — Jacob Lee Suchardt, Hana El-Shazli &
> Pierluigi Cassotti, EMNLP 2025.
> 📄 https://aclanthology.org/2025.emnlp-main.1600/

**The gist.** "STIPA" = *Speech-to-IPA*: transcribing speech directly into narrow
IPA, as a language-agnostic tool for documenting under-resourced and unwritten
languages (where no orthography or pronunciation dictionary exists). The paper
fine-tunes OpenAI Whisper for IPA output (their **WhIPA** / **LoWhIPA** models),
and benchmarks it against existing universal phone recognizers — most notably
**MultIPA** — across **21 language families**, scoring with **PER** and **PFER**.
Their headline finding: fine-tuned Whisper reaches state-of-the-art on *seen*
languages, while MultIPA remains the most robust on *unseen* ones; PER and PFER
often diverge because PER is a harsher exact-match metric.

**What we reuse.** The two metrics, the [VoxAngeles](https://github.com/pacscilab/voxangeles)
corpus (audited narrow IPA, the same 21 families), and several of the same
models. **What we add:** five additional public models the paper does not score
(Allosaurus, Wav2Vec2Phoneme, Allophant, ZIPA, POWSM), all run on CPU. This lets
us (a) sanity-check our pipeline against the paper's published WhIPA/MultIPA
numbers, then (b) place the newer models on the same scale.

[↑ Back to top](#table-of-contents)

---

## 3. Models evaluated

Each model runs in its own environment (see [§7](#7-setup--automatic-vs-manual)).
"Config name" is the `--model`/`--variant` you pass to `run_inference.py`.

| Config name | Model | Paper | Code / Weights |
|---|---|---|---|
| `allosaurus` | Allosaurus (universal phone recognizer) | [arXiv:2002.11800](https://arxiv.org/abs/2002.11800) | [github/xinjli/allosaurus](https://github.com/xinjli/allosaurus) |
| `wav2vec2phoneme` | Wav2Vec2Phoneme (XLSR-53, espeak-CV-ft) | [arXiv:2109.11680](https://arxiv.org/abs/2109.11680) | [hf/facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) |
| `multipa` | MultIPA (Taguchi & Sakai) | [arXiv:2308.03917](https://arxiv.org/abs/2308.03917) | [github/ctaguchi/multipa](https://github.com/ctaguchi/multipa) · [hf model](https://huggingface.co/ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns) |
| `allophant` | Allophant (feature-composition) | [arXiv:2306.04306](https://arxiv.org/abs/2306.04306) | [hf/kgnlp/allophant](https://huggingface.co/kgnlp/allophant) |
| `zipa` (small) | ZIPA-small (Zipformer, ONNX) | [arXiv:2505.23170](https://arxiv.org/abs/2505.23170) | [github/lingjzhu/zipa](https://github.com/lingjzhu/zipa) · [hf weights](https://huggingface.co/anyspeech/zipa-small-crctc-ns-700k) |
| `zipa_large` | ZIPA-large (Zipformer, ONNX) | [arXiv:2505.23170](https://arxiv.org/abs/2505.23170) | [hf weights](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k) |
| `powsm` | POWSM (ESPnet S2T, `<pr>` task) | [arXiv:2510.24992](https://arxiv.org/abs/2510.24992) | [hf/espnet/powsm](https://huggingface.co/espnet/powsm) |
| `whipa` (base-cv) | WhIPA Base (full fine-tune, CV) | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1600/) | [github/jshrdt/whipa](https://github.com/jshrdt/whipa) · [hf weights](https://huggingface.co/jshrdt/whipa-base-cv) |
| `whipa_comb` (base-comb) | LoWhIPA Base (LoRA, Combined) | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1600/) | [hf weights](https://huggingface.co/jshrdt/lowhipa-base-comb) |

> WhIPA variants here use **whisper-base** (74M) for CPU speed; the paper's
> headline results use **whisper-large-v2**. So our WhIPA rows are honest but on
> the smaller backbone — see [§4](#4-results-our-numbers-vs-the-paper).

[↑ Back to top](#table-of-contents)

---

## 4. Results: our numbers vs. the paper

All values are **%** (PER and PFER ×100). Our numbers below are from the
**balanced run** (`runs/balanced/`): 9 families × 2 langs × 40 utts = 720 files.
The paper's column is its **VoxAngeles "Overall"** result (Table 3), computed over
all 21 families. The sets differ (ours is balanced across families; theirs spans
the full corpus), so treat this as a **calibration check**, not an identical
re-run — the full-corpus run (`runs/full/`, see [§11](#11-reproducing--verifying-the-numbers))
is the apples-to-apples match and will be updated here when it finishes.

### Models also evaluated in the paper (direct comparison)

| Config | = Paper model | Our PER | Paper PER | Our PFER | Paper PFER |
|---|---|--:|--:|--:|--:|
| `multipa` | MultIPA | 68.9 | 60.1 | 17.9 | 15.4 |
| `whipa` | CV WhIPA Base | 87.7 | 87.8 | 30.1 | 32.1 |
| `whipa_comb` | Combined LoWhIPA Base | 83.1 | 81.6 | 23.8 | 23.0 |

The overlapping models track the paper closely — WhIPA Base lands almost exactly
on the paper's number (87.7 vs. 87.8 PER), and LoWhIPA within ~1 point — a good
sign the pipeline and metrics are faithful. MultIPA reads a few points harder
here because our balanced set deliberately includes difficult unseen families.

### Models we add (not scored on VoxAngeles in the paper)

Sorted best-PER first (720-file balanced run):

| Config | Our PER | Our PFER |
|---|--:|--:|
| `powsm` | 60.8 | 19.0 |
| `wav2vec2phoneme` | 61.5 | 20.6 |
| `multipa` | 68.9 | 17.9 |
| `zipa` (small) | 70.2 | 20.2 |
| `allophant` | 73.2 | 22.9 |
| `zipa_large` | 73.6 | 21.8 |
| `whipa_comb` | 83.1 | 23.8 |
| `whipa` | 87.7 | 30.1 |
| `allosaurus` | 93.3 | 33.1 |

On this set POWSM leads on both metrics. **High PER with low PFER is expected**,
not a bug: the references are *narrow* audited IPA (length marks, devoicing,
dentals) while models emit *broad* phones — exact-match PER punishes `iː`≠`i`,
but feature-based PFER stays low because the phones are articulatorily close.

Full per-family breakdowns: `runs/<name>/summary_by_family.tsv`; reference
snapshot to verify against: `runs/PREVIOUS_RESULTS.md`.

[↑ Back to top](#table-of-contents)

---

## 5. How the evaluation works

The pipeline is three stages; each writes files the next stage reads, so you can
stop and inspect between them.

**Stage 1 — Build (`scripts/build_testset.py` or `build_full.py`).**
Reads the VoxAngeles corpus, picks utterances, resamples each to 16 kHz mono WAV
into `audio/`, segments the audited IPA into space-separated phones with
[panphon](https://github.com/dmort27/panphon), and writes `references.tsv`
(`filename, language, family, reference_ipa`). The `<family>_<lang>_<NNN>.wav`
naming is what lets the scorer group results by language family.

**Stage 2 — Inference (`scripts/run_inference.py --model X`).**
Loads one model, reads every `audio/*.wav`, and writes its IPA guess to
`results/X/<clip>.txt` (one line per clip). Each model is run with **its own
environment's Python** because their dependencies conflict; the script imports
each model's libraries lazily so only the active model's deps must be present.

**Stage 3 — Score (`scripts/evaluate.py`).**
For every clip × model it computes:
- **PER** — edit distance over phones ÷ reference length (via `jiwer`); phones
  must be space-separated, hence the segmentation in Stage 1.
- **PFER** — panphon feature edit distance ÷ number of reference phones;
  substitution cost scales with how many articulatory features differ, so a
  near-miss costs less than a wrong phone.
It writes `per_file_results.tsv` (every clip), `summary_by_model.tsv` (mean per
model), and `summary_by_family.tsv` (mean per model per family).

[↑ Back to top](#table-of-contents)

---

## 6. Repository layout

```
phonetic_eval/
├── scripts/
│   ├── build_testset.py        # build a BALANCED subset (N fam × K lang × M utt)
│   ├── build_full.py           # build the ENTIRE corpus (every usable utterance)
│   ├── run_inference.py        # run one model over audio/  ->  results/<model>/
│   ├── evaluate.py             # score results/ against references.tsv -> *.tsv
│   ├── run_all.sh              # run all 9 models over the current audio/
│   ├── run_dataset.sh          # build + run + score one named dataset -> runs/<name>/
│   └── run_full_pipeline.sh    # balanced run, then full-corpus run, unattended
├── runs/                       # RESULTS folder (everything scored lives here)
│   ├── PREVIOUS_RESULTS.md     #   reference numbers to verify against
│   └── <name>/                 #   archived summaries per dataset (160, balanced, full):
│                               #   summary_by_model.tsv, summary_by_family.tsv,
│                               #   per_file_results.tsv, references.tsv
├── audio/        (gitignored)  # generated WAVs
├── corpora/      (gitignored)  # VoxAngeles + cloned model repos
├── envs/         (gitignored)  # per-model environments
├── models/       (gitignored)  # downloaded ZIPA/WhIPA weights
└── RESUME.md                   # detailed working notes / history
```

Large or license-restricted directories (`audio/`, `corpora/`, `envs/`,
`models/`, `papers/`, `results/`) are **not** committed — they are regenerated by
the steps below. The scorer also writes transient working copies
(`references.tsv`, `summary_by_model.tsv`, `summary_by_family.tsv`,
`per_file_results.tsv`) to the project root; these are gitignored because each
run overwrites them — the kept-per-dataset copies live under `runs/<name>/`.

[↑ Back to top](#table-of-contents)

---

## 7. Setup — automatic vs. manual

You need Python 3.10–3.12, plus `ffmpeg`, `git`, and `tmux`. There is no install
script that builds *every* model environment for you (they intentionally conflict),
so setup is done per environment. The two paths below differ in **how much you do
by hand**.

### 7a. Prerequisite tools (no root needed)

If `ffmpeg`/`git`/`tmux` aren't available system-wide, install them without root
using [micromamba](https://mamba.readthedocs.io/) — a single static binary:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p ~/.local/bin && mv bin/micromamba ~/.local/bin/
~/.local/bin/micromamba create -y -p ./envs/tools -c conda-forge ffmpeg git tmux
export PATH="$PWD/envs/tools/bin:$PATH"
```

### 7b. Per-model environments

Create only the environments for the models you intend to run. Each block makes
one isolated env and installs that model's stack.

```bash
# scoring (ALWAYS needed): jiwer + panphon
python -m venv envs/env_score && source envs/env_score/bin/activate
pip install jiwer panphon pandas && deactivate

# allosaurus  (model 'uni2005' auto-caches on first run)
python -m venv envs/env_allosaurus && source envs/env_allosaurus/bin/activate
pip install allosaurus torch soundfile && deactivate

# HF models share one env: wav2vec2phoneme, multipa, whipa (base-cv + base-comb)
python -m venv envs/env_hf && source envs/env_hf/bin/activate
pip install torch torchaudio transformers librosa soundfile peft panphon pandas pyyaml
deactivate

# allophant  (built with uv; this env ships no pip, so use `uv pip`)
uv venv -p 3.11 envs/env_allophant
uv pip install --python envs/env_allophant/bin/python allophant soundfile

# zipa  (ONNX inference path — avoids the k2/icefall deps the torch path needs)
python -m venv envs/env_zipa && source envs/env_zipa/bin/activate
pip install onnxruntime torch soundfile librosa lhotse && deactivate

# powsm  (ESPnet). pyworld has no wheel here and there's no gcc, so use micromamba
# to pull a prebuilt pyworld + compilers, then pip the rest into that env:
~/.local/bin/micromamba create -y -p ./envs/env_powsm -c conda-forge \
    python=3.10 pyworld "numpy<2" compilers
envs/env_powsm/bin/pip install espnet espnet_model_zoo torch torchaudio soundfile librosa
```

### 7c. Model repos + weights

```bash
mkdir -p corpora models
# ZIPA: clone the repo (ONNX helper) and place model.onnx + tokens.txt under models/
git clone https://github.com/lingjzhu/zipa corpora/zipa_repo
#   models/zipa_small_ns/   <- huggingface.co/anyspeech/zipa-small-crctc-ns-700k
#   models/zipa_large_ns/   <- huggingface.co/anyspeech/zipa-large-crctc-ns-800k

# WhIPA: clone the repo (loader class) and place weights under models/
git clone https://github.com/jshrdt/whipa corpora/whipa_repo
#   models/whipa_base_cv/       <- huggingface.co/jshrdt/whipa-base-cv
#   models/lowhipa_base_comb/   <- huggingface.co/jshrdt/lowhipa-base-comb  (LoRA)
```

POWSM (`espnet/powsm`) and `openai/whisper-base` download automatically on first
run into `~/.cache/huggingface`.

### 7d. Corpus

Fetch VoxAngeles into `corpora/voxangeles/`:

```bash
git clone https://github.com/pacscilab/voxangeles corpora/voxangeles
```

It provides audited IPA (`transcriptions/voxangeles_transcriptions.tsv`,
`updated` column), per-language audio (`data/audited_aligned/<lang>.zip`), and
family maps (`lrec-coling_analyses/map/`).

[↑ Back to top](#table-of-contents)

---

## 8. Building the test set

The scorer reads `audio/*.wav` + `references.tsv`. Two builders generate both.
**Each build overwrites `audio/` and `references.tsv`**, so only one test set is
"live" at a time. Builders need panphon (`env_score`) and ffmpeg.

```bash
source envs/env_score/bin/activate
export FFMPEG=envs/tools/bin/ffmpeg     # or just have ffmpeg on PATH

# (a) BALANCED subset — N families × K langs per family × M utterances each.
#     Priority families are filled first; families that can't fill the quota are
#     skipped with a warning. This is the recommended set for clean comparison.
python scripts/build_testset.py --families 11 --langs-per-family 2 --utts 40

# (b) FULL corpus — every usable utterance in every language (~5,400 files).
python scripts/build_full.py
deactivate
```

Both write `<family>_<lang>_<NNN>.wav` and a matching row in `references.tsv`.

### Hand-made test files

To skip the builders and supply your own clips: drop 16 kHz mono WAVs in `audio/`
and add one row per clip to `references.tsv` (tab-separated, phones
space-separated):

```
filename	language	family	reference_ipa
myclip_001.wav	Finnish	Uralic	t ä m ä
```

Resample anything else with
`ffmpeg -i in.wav -ar 16000 -ac 1 audio/myclip_001.wav`.

[↑ Back to top](#table-of-contents)

---

## 9. Running inference

### 9a. Automatic — everything via one script

Runs all nine configs over the **current** `audio/` set, each in its own env:

```bash
bash scripts/run_all.sh            # logs to results/run_all.log, ~15 min on the pilot
```

For the **full unattended workflow** (build → run → score the balanced set, then
do the same for the full corpus, archiving each to `runs/<name>/`), use the
pipeline driver inside `tmux` so it survives an SSH drop or logoff:

```bash
envs/tools/bin/tmux new-session -d -s phoneval 'bash scripts/run_full_pipeline.sh'
cat runs/STATUS                    # top-level progress, without attaching
envs/tools/bin/tmux attach -t phoneval     # watch live; Ctrl-b then d to detach
```

### 9b. Manual — one model at a time

Run `run_inference.py` with **that model's own interpreter**. It reads every WAV
in `audio/` and writes `results/<model>/<clip>.txt`. This is the way to test or
debug a single model.

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

What to know per model:
- **`--variant`** is required for `zipa` (`small`/`large`) and `whipa`
  (`base-cv`/`base-comb`); the others take no variant.
- Results subfolders: `zipa`→`results/zipa/`, `zipa_large`→`results/zipa_large/`,
  `whipa base-cv`→`results/whipa/`, `whipa base-comb`→`results/whipa_comb/`.
- **`powsm`** is the slowest on CPU (~2 s/clip) and conditions on the ISO-639-3
  code embedded in each filename.
- After a run, eyeball a few outputs: `head results/multipa/*.txt`.

[↑ Back to top](#table-of-contents)

---

## 10. Scoring (PER & PFER)

Once `results/<model>/` folders exist, score them all at once:

```bash
envs/env_score/bin/python scripts/evaluate.py
column -t -s$'\t' summary_by_model.tsv
```

`evaluate.py` matches each `results/<model>/<clip>.txt` to its `references.tsv`
row and writes three TSVs (per clip, per model, per family). It normalizes
Unicode and strips non-phone tokens before scoring; see the metric notes in
[§5](#5-how-the-evaluation-works) and the cleaning logic in the script's
`clean_ipa()`.

[↑ Back to top](#table-of-contents)

---

## 11. Reproducing & verifying the numbers

To confirm a model still reproduces `runs/PREVIOUS_RESULTS.md`:

1. Build the **160-file** pilot: `python scripts/build_testset.py --families 8
   --langs-per-family 2 --utts 10`.
2. Run that model (see [§9b](#9b-manual--one-model-at-a-time)) and re-score
   ([§10](#10-scoring-per--pfer)).
3. Compare its row to the table in `runs/PREVIOUS_RESULTS.md`.

Archived runs live in `runs/<name>/` (each with its own `references.tsv` and
summaries). The **full-corpus** run (`runs/full/`) is the set directly comparable
to the paper's Table 3; when it completes, its `summary_by_model.tsv` replaces the
calibration numbers in [§4](#4-results-our-numbers-vs-the-paper).

A high PER with a low PFER is **expected**, not a regression — it is the
broad-vs-narrow IPA mismatch described in [§4](#4-results-our-numbers-vs-the-paper)
and `RESUME.md`.

[↑ Back to top](#table-of-contents)

---

## 12. Notes, gotchas & license

- **CPU only.** Every model is configured for CPU. WhIPA uses whisper-base (not
  the paper's whisper-large-v2) for speed; expect weaker WhIPA numbers than the
  paper's headline as a result.
- **One test set at a time.** Builders overwrite `audio/` + `references.tsv`.
  Always read *finished* numbers from `runs/<name>/`, not the project root, if a
  multi-phase run is in progress.
- **Why separate environments.** The models pin conflicting versions of PyTorch,
  transformers, ONNX runtime, and k2; `run_inference.py` imports each model's
  libraries lazily so each runs under its own interpreter.
- **Known fixes** baked into `run_inference.py` (documented inline there):
  wav2vec2phoneme loaded with `do_phonemize=False` to avoid the espeak
  dependency; WhIPA's buggy `transcribe_ipa()` reimplemented; ZIPA decoded via
  ONNX to avoid k2/icefall; POWSM's slash-wrapped output split into phones.
- **Data/licensing.** VoxAngeles and the model weights are **not** redistributed
  here — fetch them from their upstreams ([§7](#7-setup--automatic-vs-manual)).
  See `RESUME.md` for the full working history and per-model details.

[↑ Back to top](#table-of-contents)
