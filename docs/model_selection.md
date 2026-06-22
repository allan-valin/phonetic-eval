# Model & checkpoint selection — rationale

Paper-ready justification for **which checkpoint we used for each model, and why we
did *not* test the available alternatives.** Every comparative number here is from
the model's own paper (in `papers/`); table references are to those papers. The
short version: we use each model's **best released, most-multilingual** checkpoint,
and we decline to re-run alternatives that the source papers already show differ
negligibly, that our folded metric already captures, or that were never published.

## Selection principles

1. **Best released, most-multilingual checkpoint per model.** A 21-family benchmark
   should see each system at its strongest and broadest.
2. **Don't re-run near-identical variants.** Where a paper's own table shows two
   variants within ~1 point, testing both buys one redundant table row for hours of
   CPU. Cite the paper instead.
3. **Don't re-run what our metrics already isolate.** The folded PER/PFER variant
   ([README §10](../README.md#10-scoring-per--pfer)) already separates broad-vs-narrow
   reference effects, so a separately-trained "broad" checkpoint adds little.
4. **Can't test what wasn't released.** Some paper variants have no public weights.
5. **Do test an axis when it is real and non-redundant** — e.g. the WhIPA
   whisper-base → whisper-large backbone, which we *did* add (`whipa_large`) because
   it moved results substantially (strict PER −14 VoxAngeles / −24 FLEURS) and no
   other run covered it.

## Per-model

### Allosaurus (Li et al., ICASSP 2020)
**Used:** the package's default universal recognizer (`uni2005`). It is the single
public universal checkpoint; the paper's PHOIBLE-augmented decoding improves *unseen*
languages but is a decoding option, not a separate released model. No alternative to
weigh.

### Wav2Vec2Phoneme (Xu et al., Interspeech 2022)
**Used:** `facebook/wav2vec2-xlsr-53-espeak-cv-ft`.
**Alternative released:** `facebook/wav2vec2-lv-60-espeak-cv-ft`.
xlsr-53 is pretrained on **53 languages**; lv-60 on English LibriVox-60k
(monolingual). The ZIPA paper benchmarks both on the same multilingual seen set
(its Table 2, average PFER): **xlsr-53 = 11.88** vs **lv-60 = 14.36** — xlsr-53 is
~2.5 points (~17% rel.) better. We therefore use xlsr-53 as the stronger, more
multilingual representative; lv-60 would mainly re-confirm that English-centric
pretraining transfers worse cross-lingually — an expected result, not a new finding.

### MultIPA (Taguchi & Sakai, Interspeech 2023)
**Used:** `ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns` (1k training
samples/language, noisy-student).
**Paper variants:** 1k / 2k / full samples per language — Table 3 (supervised
"Overall"): PER **24.9 / 22.4 / 21.0**, PFER **7.0 / 6.2 / 5.7** (a real ~4-point
trend). **Only the 1k-ns checkpoint is published on the Hub**; the 2k and full
models were never released, so we cannot run them. Documented by necessity.

### Allophant (Glocker et al., Interspeech 2023)
**Used:** `kgnlp/allophant` (Multi-Task).
**All five paper models are on the Hub** (`allophant`, `-shared`, `-hierarchical`,
`-baseline`, `-baseline-shared`). Paper Table 1 (zero-shot UCLA PER):

| variant | PER | AER |
|---|--:|--:|
| Baseline | 57.01 | – |
| Baseline Shared | 48.25 | – |
| Multi-Task Shared | 46.05 | 19.52 |
| **Multi-Task (used)** | **45.62** | **19.44** |
| Multi-Task Hierarchy | 46.09 | 19.18 |

The three multi-task variants sit within **0.5 PER points** of each other — testing
all three is a flat line. The only non-trivial contrast is **Baseline → Multi-Task
(~11 points)**, which is Allophant's central feature-composition claim; that single
ablation (`allophant-baseline`) is the only alternative here that would *show*
anything, and is the one to add if any. We use Multi-Task as the paper's best model.

### ZIPA (Zhu et al., ACL 2025)
**Used:** `zipa-small-crctc-ns-700k` (64M) and `zipa-large-crctc-ns-800k` (300M) —
the noisy-student CR-CTC final averaged checkpoints (ONNX).
**Alternatives (all ONNX-available):** transducer (`T`), earlier iterations
(300k/500k), causal/noncausal, and **no-diacritics**. From the paper's Table 2
(seen, average PFER):

- **Transducer ≈ CTC:** ZIPA-T-LARGE-500k **2.70** vs ZIPA-CR-NS-LARGE-800k **2.71**
  — indistinguishable.
- **Iterations are monotonic** (CR-SMALL 300k **9.08** → 500k **5.62** → ns-700k
  **3.22**): a training curve, not a model-selection axis; we take the final
  checkpoint.
- **No-diacritics barely moves** the paper's own numbers (CR-NS-SMALL **3.22→3.02**,
  LARGE **2.71→2.65**), and — more importantly — **our folded metric already isolates
  the diacritic/broad-vs-narrow effect** for *every* model uniformly, so a separately
  broad-trained ZIPA would be redundant with that analysis.

We use CR-NS (the paper's best configuration) at the final checkpoint; the rest add
no separable signal on our setup.

### POWSM (2025)
**Used:** `espnet/powsm` — the attention encoder-decoder (AED), phone-recognition
task `<pr>` (the paper's main model; trained with a hybrid CTC/attention objective,
α_ctc = 0.3, decoded via attention).
**Alternative released:** `espnet/powsm_ctc` (CTC-only head). This is an architecture
ablation requiring a different inference path; the paper's headline phone-recognition
results are the AED, which we use.

### WhIPA / LoWhIPA (Suchardt, El-Shazli & Cassotti, EMNLP 2025) — the reference paper
**Used:** `whipa` (whipa-base-cv, full fine-tune), `whipa_comb` (lowhipa-base-comb,
LoRA), and `whipa_large` (whipa-large-cv, full fine-tune, whisper-large-v2).

The author collection also releases LoWhIPA adapters that differ by **training-data
language** (`-cv` Common Voice, `-asc` South-Levantine Arabic, `-thchs30` Mandarin,
`-comb` combined, plus large `-sr`). Those probe *training-data composition*, an axis
our **21-family** coverage already exercises broadly, so running each
language-specialized adapter adds little to a multilingual leaderboard.

The one WhIPA axis we *did* add is the **backbone** (`whipa_large`, whisper-base →
whisper-large-v2), because it is the WhIPA variable that produced a large,
non-redundant effect on our data (strict PER −14.0 on VoxAngeles balanced, −24.2 on
FLEURS; see [README §4](../README.md#4-results-recomputed-numbers-vs-the-stipa-paper)).
This matches the paper's emphasis on backbone size and training-data composition.

## Summary table

| Model | Checkpoint used | Best alternative not run | Reason not run |
|---|---|---|---|
| Allosaurus | `uni2005` (default) | — | only public universal checkpoint |
| Wav2Vec2Phoneme | xlsr-53-espeak-cv-ft | lv-60-espeak-cv-ft | English-centric, worse multiling. (11.88 vs 14.36 PFER) |
| MultIPA | xlsr-…-ipa1000-ns (1k) | 2k / full | **not released** (paper: 24.9 / 22.4 / 21.0 PER) |
| Allophant | Multi-Task | baseline (−11 pp), shared/hierarchy (±0.5 pp) | shared/hierarchy near-identical; baseline is the only informative one |
| ZIPA | CR-NS small/large (final) | T, iters, no-diacritics | T≈CTC (2.70/2.71); iters monotonic; no-diacritics ≈ folded metric |
| POWSM | AED (`<pr>`) | powsm_ctc | architecture ablation, separate inference path |
| WhIPA/LoWhIPA | base-cv, base-comb, **large-cv** | LoWhIPA `-asc`/`-thchs30`/`-sr` | training-language variants already covered by 21-family breadth |
