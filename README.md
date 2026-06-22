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
4. [Results: recomputed numbers vs. the STIPA paper](#4-results-recomputed-numbers-vs-the-stipa-paper)
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
"Config name" is the `--model`/`--variant` you pass to `run_inference.py`. **Rows
are ordered by paper publication date** (oldest first); §4 uses the same order.

| Config name | Model | Published | Paper | Code / Weights |
|---|---|--:|---|---|
| `allosaurus` | Allosaurus (universal phone recognizer) | 2020 | [arXiv:2002.11800](https://arxiv.org/abs/2002.11800) (ICASSP 2020) | [github/xinjli/allosaurus](https://github.com/xinjli/allosaurus) |
| `wav2vec2phoneme` | Wav2Vec2Phoneme (XLSR-53, espeak-CV-ft) | 2021 | [arXiv:2109.11680](https://arxiv.org/abs/2109.11680) (Interspeech 2022) | [hf/facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) |
| `allophant` | Allophant (feature-composition) | 2023-06 | [arXiv:2306.04306](https://arxiv.org/abs/2306.04306) (Interspeech 2023) | [hf/kgnlp/allophant](https://huggingface.co/kgnlp/allophant) |
| `multipa` | MultIPA (Taguchi & Sakai) | 2023-08 | [arXiv:2308.03917](https://arxiv.org/abs/2308.03917) (Interspeech 2023) | [github/ctaguchi/multipa](https://github.com/ctaguchi/multipa) · [hf model](https://huggingface.co/ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns) |
| `zipa` (small) | ZIPA-small (Zipformer, ONNX) | 2025-05 | [arXiv:2505.23170](https://arxiv.org/abs/2505.23170) (ACL 2025) | [github/lingjzhu/zipa](https://github.com/lingjzhu/zipa) · [hf weights](https://huggingface.co/anyspeech/zipa-small-crctc-ns-700k) |
| `zipa_large` | ZIPA-large (Zipformer, ONNX) | 2025-05 | [arXiv:2505.23170](https://arxiv.org/abs/2505.23170) (ACL 2025) | [hf weights](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k) |
| `powsm` | POWSM (ESPnet S2T, `<pr>` task) | 2025-10 | [arXiv:2510.24992](https://arxiv.org/abs/2510.24992) | [hf/espnet/powsm](https://huggingface.co/espnet/powsm) |
| `whipa` (base-cv) | WhIPA Base (full fine-tune, CV) | 2025-11 | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1600/) | [github/jshrdt/whipa](https://github.com/jshrdt/whipa) · [hf weights](https://huggingface.co/jshrdt/whipa-base-cv) |
| `whipa_comb` (base-comb) | LoWhIPA Base (LoRA, Combined) | 2025-11 | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1600/) | [hf weights](https://huggingface.co/jshrdt/lowhipa-base-comb) |

> WhIPA variants here use **whisper-base** (74M) for CPU speed; the paper's
> headline results use **whisper-large-v2**. So our WhIPA rows are honest but on
> the smaller backbone — see [§4](#4-results-recomputed-numbers-vs-the-stipa-paper).

[↑ Back to top](#table-of-contents)

---

## 4. Results: recomputed numbers vs. the STIPA paper

The reference benchmark throughout this section is the **STIPA paper** (Suchardt,
El-Shazli & Cassotti, EMNLP 2025; [see §2](#2-the-paper-this-is-based-on)) — *not* the
individual model papers linked in [§3](#3-models-evaluated). All values are **%**
(PER and PFER ×100). The **Recomputed** columns are our own re-run of each model on
VoxAngeles (this repo); the **STIPA paper** columns are that paper's published
VoxAngeles "Overall" result (its Table 3).

Recomputed numbers below are from the **full-corpus run** (`runs/full/`): all 21
families, 95 languages, 5416 files — the apples-to-apples match for the STIPA
paper's "Overall", which is also computed over the full corpus. A **balanced run**
(`runs/balanced/`: 9 families × 2 langs × 40 utts = 720 files) is kept as a
cross-check; its numbers read a few points harder because it deliberately
over-weights difficult unseen families. See
[§11](#11-reproducing--verifying-the-numbers) for how both were produced.

### Each model: as published vs. on our VoxAngeles

This is the headline comparison and the most useful one to read carefully. Each
model's **own paper** reports a low error rate — but on its *own* benchmark, which
is typically the model's **seen / in-domain / broadly-transcribed** test data (the
favourable case). Our columns re-run the same model on **narrow, audited VoxAngeles
IPA across 21 families** (the hard case). The numbers are **not directly
comparable** — different test languages, different reference granularity — and that
is exactly the point: it shows how far a "good benchmark" number travels when you
move to a hard, narrow, broad-coverage one.

Rows are **ordered by publication date**. Reported numbers are the headline figure
from each paper (table cited). Our VoxAngeles columns show **strict→folded** (full
corpus; folding strips diacritics/length/tone, see [§10](#10-scoring-per--pfer)).
All values are **%** (PER and PFER ×100); lower is better.

| Config | Yr | Reported PER | Reported PFER | …on its own benchmark | Our VoxAngeles PER | Our VoxAngeles PFER |
|---|--:|--:|--:|---|--:|--:|
| `allosaurus` | 2020 | 25.0 | — | 11-lang multiling., *seen* (Li T1) | 91.7→82.6 | 32.7→32.5 |
| `wav2vec2phoneme` | 2021 | 22.2 | — | CommonVoice 13-lang, x-ling (Xu T8) | 53.0→41.7 | 15.4→13.6 |
| `allophant` | 2023 | 45.6 | 19.4 † | UCLA, zero-shot (Glocker T1) | 71.0→50.6 | 20.5→18.6 |
| `multipa` | 2023 | 21.0 | 5.7 | CommonVoice, *supervised* (Taguchi T3) | 62.4→50.3 | 15.0→14.1 |
| `zipa` (small) | 2025 | — | 3.2 ‡ | IPAPack++, *seen* (Zhu T2) | 64.2→50.4 | 18.0→15.9 |
| `zipa_large` | 2025 | — | 2.7 ‡ | IPAPack++, *seen* (Zhu T2) | 69.1→54.5 | 20.3→18.1 |
| `powsm` | 2025 | — | 2.6 ‡ | IPAPack++, in-domain (POWSM T2) | 55.5→45.7 | 16.1→15.1 |
| `whipa` | 2025 | 87.8 | 32.1 | **VoxAngeles** (STIPA T3) | 83.7→68.8 | 27.1→25.4 |
| `whipa_comb` | 2025 | 81.6 | 23.0 | **VoxAngeles** (STIPA T3) | 77.8→55.4 | 19.3→17.1 |

† Allophant's paper reports **AER** (articulatory-attribute error rate), a
feature-level error related to but not identical to PFER.
‡ ZIPA and POWSM report **PFER** in the same panphon feature-distance family we use,
so the scale *is* comparable — which makes the contrast stark: the same `zipa`
small model is ~3.2 on its in-domain seen set but ~18 on our narrow VoxAngeles,
~6× the error purely from benchmark difficulty and reference granularity.

Read the table left-to-right per row: a model that looks near-solved on its home
turf (MultIPA 21.0 PER supervised; ZIPA/POWSM PFER ~3) lands far higher on narrow
multi-family VoxAngeles. The only rows on the **same** benchmark as us are
`whipa`/`whipa_comb` (the STIPA paper also used VoxAngeles) — and there our
recompute matches their published numbers within ~4 points (see calibration below).

### Calibration: models the STIPA paper also ran on VoxAngeles

These three are the **apples-to-apples** check — same benchmark, so the columns
*are* comparable. Our recompute tracking the STIPA paper closely is the evidence
that the pipeline and metrics are faithful.

| Config | = STIPA-paper model | Recomputed PER | STIPA-paper PER | Recomputed PFER | STIPA-paper PFER |
|---|---|--:|--:|--:|--:|
| `multipa` | MultIPA | 62.4 | 60.1 | 15.0 | 15.4 |
| `whipa` | CV WhIPA Base | 83.7 | 87.8 | 27.1 | 32.1 |
| `whipa_comb` | Combined LoWhIPA Base | 77.8 | 81.6 | 19.3 | 23.0 |

MultIPA lands within ~2 PER points and near-exact on PFER; WhIPA Base and LoWhIPA
within ~4. (Note MultIPA's *own-paper* number above — 21.0 PER on supervised
CommonVoice — is far lower than its 60.1 here: same model, easier benchmark.)

### Strict vs. folded, and why PER ≫ PFER

**What "strict" and "folded" mean** (the two scoring modes, defined precisely in
[`scripts/evaluate.py`](scripts/evaluate.py)):

- **Strict** — compare the predicted phones to the reference *exactly as written*.
  Each IPA symbol is matched as a whole Unicode grapheme, so a length mark or
  diacritic is part of the token: `iː` ≠ `i`, `b̥` ≠ `b`, `t̪` ≠ `t` are all full
  mismatches. This is the harsh, literal metric.
- **Folded** — first normalise both prediction and reference to *broad* phones by
  deleting the diacritic/length/tone marks (concretely: decompose to Unicode NFD,
  then drop all code points in categories `Mn`/`Lm`/`Sk` — combining marks,
  modifier letters, tone-bar symbols), *then* compare. So `iː`→`i` and `b̥`→`b`
  line up. This measures "did the model get the base phone right, ignoring fine
  phonetic detail."

The same two error metrics are computed in each mode — **PER** (phone error rate:
Levenshtein edit distance over phones ÷ reference length, an exact token match) and
**PFER** (phonological-feature error rate: panphon's feature-weighted edit distance
÷ reference phones, so a substitution costs *less* the more articulatory features
the two phones share). PER is all-or-nothing per phone; PFER is graded.

**Worked example** — Icelandic `indoeuropean_isl_001.wav`, reference **`b iː ð`**.
Values are % (lower is better); `strict → folded`:

| Config | hypothesis | PER (strict→folded) | PFER (strict→folded) |
|---|---|--:|--:|
| `allosaurus` | `b̥ i ð` | 66.7 → 0.0 | 1.4 → 0.0 |
| `wav2vec2phoneme` | `b iː θ` | 33.3 → 33.3 | 1.4 → 1.4 |
| `allophant` | `b iɪ d̪` | 66.7 → 66.7 | 33.3 → 33.3 |
| `multipa` | `piθ` | 100.0 → 66.7 | 4.2 → 2.8 |
| `zipa` (small) | `p i t s` | 133.3 → 100.0 | 38.2 → 36.8 |
| `zipa_large` | `b i θ` | 66.7 → 33.3 | 2.8 → 1.4 |
| `powsm` | `p e iː θ` | 100.0 → 100.0 | 33.3 → 33.3 |
| `whipa` | `biz` | 66.7 → 33.3 | 4.2 → 2.8 |
| `whipa_comb` | `piːtː` | 66.7 → 66.7 | 6.9 → 5.6 |

Read `allosaurus`: it heard `b̥ i ð` — phonetically a near-perfect match, but
strict PER is **66.7%** because `b̥`≠`b` and `i`≠`iː` count as two wrong phones.
Fold away the diacritic and length mark and PER drops to **0.0%**; PFER was already
~0 because those phones are one feature apart. That single row is the whole story
of this benchmark in miniature. (Note `multipa`/`whipa`/`whipa_comb` emit no spaces,
e.g. `piθ`; the scorer segments such strings into phones with panphon before
scoring.)

**At the corpus level** `wav2vec2phoneme` leads on PER (it and POWSM swap the top
two spots between the balanced and full sets), while `multipa` edges the best PFER.
**High PER with low PFER is expected**, not a bug — the references are *narrow*
audited IPA while models emit *broad* phones. **Folding the references to broad
phones drops PER by ~11–22 points** (e.g. `whipa_comb` 77.8→55.4,
`wav2vec2phoneme` 53.0→41.7, shown in the table above) while barely moving PFER —
direct confirmation that most of the strict-PER penalty is reference granularity,
not recognition error.

### Contrast set: FLEURS + G2P (broad phonemic, connected speech)

To show how much of the VoxAngeles difficulty is its *reference style* rather than
the audio, we built a deliberately easier **contrast set** — the mirror image of
VoxAngeles on every axis:

| | VoxAngeles | FLEURS + G2P |
|---|---|---|
| speech | isolated words | full read-out **sentences** (connected) |
| reference | **gold** narrow phonetic IPA (human-audited) | **silver** broad *phonemic* IPA (Epitran G2P) |
| coverage | 21 families | 6 langs / 5 families, 120 utts |

The references here are **predicted by [Epitran](https://github.com/dmort27/epitran)**
(rule-based grapheme→phoneme, the same G2P the STIPA paper used for Common Voice) on
the FLEURS sentence transcripts — *not* ground truth. We picked orthographically
transparent languages (Spanish, German, Turkish, Finnish, Swahili, Indonesian) where
G2P is reliable. Treat this as a **benchmark-design demonstration, not a model
ranking** — it measures agreement-with-G2P on broad phones. Rows ordered by
publication date; values are % (`strict → folded`):

| Config | PER (strict→folded) | PFER (strict→folded) |
|---|--:|--:|
| `allosaurus` | 64.2 → 57.2 | 18.9 → 18.8 |
| `wav2vec2phoneme` | 36.9 → 34.6 | 10.8 → 10.5 |
| `allophant` | 49.7 → 44.9 | 16.7 → 16.1 |
| `multipa` | 47.5 → 41.6 | 13.7 → 13.6 |
| `zipa` (small) | 32.5 → 27.7 | 7.7 → 7.0 |
| `zipa_large` | 31.5 → 26.0 | 7.6 → 6.8 |
| `powsm` | **23.7 → 22.3** | **8.2 → 7.8** |
| `whipa` | 62.9 → 53.9 | 18.9 → 18.7 |
| `whipa_comb` | 55.2 → 42.6 | 14.6 → 13.9 |

Two things jump out, both confirming the reference-granularity story:

1. **PER roughly halves vs. VoxAngeles** for the same models (POWSM 55.5→23.7,
   `zipa_large` 69.1→31.5) — same models, same metric, just broad predicted refs on
   connected speech instead of narrow gold refs on isolated words.
2. **Folding barely changes the FLEURS numbers** (POWSM 23.7→22.3) whereas it moved
   VoxAngeles a lot (POWSM 55.5→45.7) — because the G2P references are *already
   broad*, there is almost no diacritic/length detail left to fold. In effect
   **folded-VoxAngeles ≈ strict-FLEURS**: both are broad-phone comparisons, and they
   land in the same range.

POWSM and ZIPA lead on this broad connected-speech set (a different ordering from
narrow VoxAngeles, where `wav2vec2phoneme` led). Reproduce with
`envs/env_hf/bin/python scripts/build_fleurs_g2p.py` then `bash scripts/run_fleurs.sh`
(archives to `runs/fleurs_g2p/`).

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

**What the code actually is.** `scripts/run_inference.py` is a thin **wrapper**:
for each model it has one function (`run_allosaurus`, `run_hf_ctc`, `run_zipa`,
`run_powsm`, `run_whipa`, `run_allophant`) that reproduces the **same call the
model's own authors document** — the snippets in [`docs/manual_runs/`](docs/manual_runs/README.md)
— plus a shared audio loader (`_load_audio_mono16k`) and a uniform output writer so
every model lands in `results/<model>/<clip>.txt` as space-separated phones. It
adds no new model: the per-model logic is exactly the upstream usage, gathered into
one file with consistent I/O and a few documented bug-fixes/output-parsing tweaks
(noted inline and in the manual docs). `scripts/evaluate.py` then scores those
folders with PER/PFER (see [§10](#10-scoring-per--pfer)). So the "automatic" path
below and the "upstream way" in `docs/manual_runs/` run the *same* model code —
one just orchestrates all nine in their own environments.

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

### 9c. Manual — the upstream way (no wrapper)

`run_inference.py` (§9b) is *this project's* convenience wrapper. To run each model
**the way its own authors document it** — install from the upstream README/model
card, prepare the audio yourself, call the author's command, read IPA off
stdout — see [`docs/manual_runs/`](docs/manual_runs/README.md): one standalone file
per model, plus a shared audio-prep section. Useful for understanding exactly what
the wrapper automates, or for reproducing a model without this repo at all.

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

**Two granularities per metric.** Each TSV reports PER and PFER twice: a *strict*
column scored on the raw narrow IPA, and a `*_folded` column scored after
`fold_to_broad()` removes every diacritic, length and tone mark (it drops all
Unicode `Mn`/`Lm`/`Sk` code points on the NFD form), collapsing e.g. `iː`→`i`,
`b̥`→`b`, `t̪`→`t`. The folded numbers are a **diagnostic** of how much PER comes
from reference granularity rather than recognition error — they do not replace the
strict metric. The gap is large for PER and small for PFER (see the [§4](#4-results-recomputed-numbers-vs-the-stipa-paper)
table), because PFER already credits articulatory closeness.

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
calibration numbers in [§4](#4-results-recomputed-numbers-vs-the-stipa-paper).

A high PER with a low PFER is **expected**, not a regression — it is the
broad-vs-narrow IPA mismatch described in [§4](#4-results-recomputed-numbers-vs-the-stipa-paper)
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
