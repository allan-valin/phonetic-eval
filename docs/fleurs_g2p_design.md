# FLEURS + G2P contrast set — design & methodology

Paper-grade documentation of the broad-phonemic contrast set, so the design can be
written up without reconstructing it from memory. Built by
[`scripts/build_fleurs_g2p.py`](../scripts/build_fleurs_g2p.py); run by
[`scripts/run_fleurs.sh`](../scripts/run_fleurs.sh); archived in `runs/fleurs_g2p/`.

## 1. Purpose

The main benchmark (VoxAngeles) pairs **isolated words** with **gold, narrow,
human-audited** phonetic IPA. High error rates there could reflect either (a) hard
audio or (b) the narrow reference style penalising broad-phone models. To separate
those two causes we built a **contrast set that inverts every variable**: connected
speech instead of isolated words, and **silver, broad, *phonemic*** references
(machine-predicted by grapheme-to-phoneme conversion) instead of gold narrow ones.
If error rates collapse on the contrast set, the reference *style* — not the
acoustics — was driving most of the VoxAngeles difficulty. They do (see §7).

This set is a **methodological probe, not a model leaderboard.** Because the
references are G2P output, the metric is *agreement with the G2P*, not agreement
with ground truth.

## 2. Speech source: FLEURS

[FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning
Evaluation of Universal Representations of Speech; Conneau et al., IEEE SLT 2022,
[arXiv:2205.12446](https://arxiv.org/abs/2205.12446)) is a 102-language read-speech
corpus: native speakers read aloud sentences from the FLoRes-101 machine-translation
benchmark (Wikipedia-derived). Properties relevant here:

- **Connected speech** — full sentences (our sample averages **12.8 s**, see §4),
  unlike VoxAngeles single words. Read, not spontaneous.
- **Native 16 kHz mono**, so no resampling/downmix is needed.
- Ships an orthographic transcript per clip in two forms: `raw_transcription`
  (cased, punctuated) and `transcription` (lowercased, punctuation stripped, numbers
  spelled out). We G2P the **normalized `transcription`**.
- **Ungated** on the HF Hub (Common Voice — the originally-planned source, and the
  one the STIPA paper used — is gated and needs an access token; FLEURS gives the
  same connected-speech + broad-G2P contrast with no token).

**Streaming, not bulk download.** We read FLEURS in `datasets` *streaming* mode and
stop after the first N valid utterances per language, so only those samples are
transferred (~0.4 MB of Hub cache total for the whole build, not the multi-GB full
corpus). Audio bytes are decoded with `soundfile` from the raw bytes
(`Audio(decode=False)`) to avoid the `torchcodec` dependency that `datasets`'
default audio decoder now requires.

## 3. Language selection

Six languages, chosen on three criteria:

1. **Orthographic transparency** — shallow grapheme→phoneme mapping, so rule-based
   G2P is reliable (we deliberately avoid deep orthographies like English/French,
   where G2P error would dominate).
2. **Epitran support** — a maintained rule set exists for the language.
3. **Family diversity** — span several families, overlapping VoxAngeles where
   possible.

| FLEURS config | Epitran code | ISO 639-3 | Language | Family |
|---|---|---|---|---|
| `es_419` | `spa-Latn` | spa | Spanish | Indo-European (Romance) |
| `de_de` | `deu-Latn` | deu | German | Indo-European (Germanic) |
| `tr_tr` | `tur-Latn` | tur | Turkish | Turkic |
| `fi_fi` | `fin-Latn` | fin | Finnish | Uralic |
| `sw_ke` | `swa-Latn` | swa | Swahili | Atlantic-Congo |
| `id_id` | `ind-Latn` | ind | Indonesian | Austronesian |

Six languages, **five families** (Spanish and German share Indo-European). The
ISO-639-3 code is embedded as the middle field of each filename
(`fleurs_<iso3>_<NNN>.wav`) because `run_inference.py` reads it there to set POWSM's
per-utterance language symbol.

## 4. Sample composition (as built)

`--utts 20` per language → **120 utterances**, ~**1532 s** (25.5 min) of audio.

| ISO | n | dur mean (s) | dur min–max (s) | phones/utt (mean) |
|---|--:|--:|--:|--:|
| deu | 20 | 12.2 | 5.2 – 23.3 | 109.8 |
| fin | 20 | 15.0 | 7.0 – 22.6 | 124.2 |
| ind | 20 | 12.1 | 5.2 – 22.6 | 111.0 |
| spa | 20 | 11.4 | 5.6 – 21.4 | 112.2 |
| swa | 20 | 14.3 | 7.8 – 24.1 | 106.5 |
| tur | 20 | 11.6 | 4.2 – 24.4 | 116.0 |
| **all** | **120** | **12.8** | **4.2 – 24.4** | **113.3** |

(References average ~113 phones/utterance — two orders of magnitude longer than a
VoxAngeles single word, which matters for the metrics: a few phone errors move PER
much less on a 113-phone sentence than on a 3-phone word.)

## 5. Reference generation (the silver labels)

References are produced by **[Epitran](https://github.com/dmort27/epitran)**
(Mortensen, Dalmia & Littell, LREC 2018) — a **rule-based** grapheme-to-phoneme
converter (per-language mapping tables + ordered post-processing rules). It is the
same G2P family the STIPA paper used to build its Common Voice references, is
pure-Python (no eSpeak/flite needed for these six languages), and emits **broad
*phonemic*** IPA — i.e. the phoneme inventory of the language, **without** the
narrow phonetic detail (length, devoicing, dentals, tone) that VoxAngeles encodes.

Pipeline per utterance:

```
FLEURS normalized transcription           "el perro corre"
        │  Epitran.transliterate()
        ▼
broad phonemic IPA (word-spaced)          "el pero kore"
        │  drop word spaces, panphon ipa_segs()   (re-segment into phones)
        ▼
space-separated phones (reference_ipa)    "e l p e r o k o r e"
```

The final re-segmentation with **panphon** (`FeatureTable.ipa_segs`) makes each
reference token exactly one IPA phone, matching the VoxAngeles reference format the
scorer expects (and discarding word boundaries, which PER does not score). See
`phone_tokenize()` in the build script.

## 6. Inference & scoring

Identical to the main benchmark: all nine model configs run unchanged (same
`run_inference.py`, each in its own environment), and scoring uses the same
[`scripts/evaluate.py`](../scripts/evaluate.py) — **strict and folded** PER and
PFER. Results archived to `runs/fleurs_g2p/{summary_by_model,summary_by_family,per_file_results}.tsv`.

Filtering during the build: an utterance is skipped if the transcription is empty,
the audio is longer than **25 s** (`MAX_SECONDS`, to bound CPU inference time), or
Epitran returns nothing segmentable.

## 7. Key results & interpretation

(Full numbers in `runs/fleurs_g2p/summary_by_model.tsv`; headline table in the main
[README §4](../README.md#4-results-recomputed-numbers-vs-the-stipa-paper).)

- **PER roughly halves vs. VoxAngeles** for the same models (e.g. POWSM 55.5→23.7,
  ZIPA-large 69.1→31.5; % strict PER). Same models, same metric — only the
  reference style and speech type changed.
- **Folding barely moves the FLEURS numbers** (POWSM 23.7→22.3 strict→folded)
  whereas it moved VoxAngeles substantially (POWSM 55.5→45.7). The G2P references
  are *already broad*, so there is almost nothing to fold away. Consequently
  **folded-VoxAngeles ≈ strict-FLEURS**: both are broad-phone comparisons and land
  in the same range. This is the cleanest single piece of evidence that
  reference granularity — not recognition error — drives the VoxAngeles PER gap.
- **Ordering changes**: POWSM and ZIPA lead on broad connected speech, whereas
  `wav2vec2phoneme` led on narrow VoxAngeles. Report orderings per-benchmark; they
  are not stable across reference styles.

## 8. Limitations / threats to validity (for the write-up)

- **Silver references.** The metric is agreement with Epitran, not ground truth.
  Absolute numbers conflate model error with G2P error and are **not comparable**
  to gold-reference benchmarks; only *within-set* and *trend* comparisons are valid.
- **Selection bias toward easy cases.** Languages were chosen *because* their
  orthography is transparent, which is exactly where G2P is most accurate — so this
  set flatters broad-phone models by construction. That is intentional (it is the
  "favourable benchmark" end of the contrast) but must be stated.
- **PFER less informative here.** With no diacritics in the references, PFER has
  little fine detail to reward; strict and folded PFER nearly coincide.
- **Read, not spontaneous, speech.** FLEURS is read Wikipedia-derived sentences;
  no disfluencies, careful pronunciation. Not representative of spontaneous or
  conversational speech.
- **G2P artefacts.** Loanwords, names, numerals and code-switched tokens can be
  mis-transcribed by rule-based G2P even in transparent orthographies.
- **Small sample.** 20 utterances × 6 languages; per-language means have wide
  confidence intervals. Treat as illustrative, not definitive.

## 9. Reproducibility

```bash
# 1) build (env_hf has datasets + epitran + soundfile + panphon)
envs/env_hf/bin/python scripts/build_fleurs_g2p.py --utts 20
# 2) run all 9 models, score, archive to runs/fleurs_g2p/
bash scripts/run_fleurs.sh
```

FLEURS streaming yields a deterministic order (parquet row order), so the same 20
utterances are selected on re-runs. The build overwrites `audio/` and
`references.tsv`; rebuild VoxAngeles afterwards with `scripts/build_full.py` (or
`build_testset.py`) if you need the main benchmark back in the working tree.

## 10. References

- **FLEURS**: Conneau, Ma, Khanuja, Zhang, Axelrod, Dalmia, Riesa, Rivera, Bapna
  (2022). *FLEURS: Few-shot Learning Evaluation of Universal Representations of
  Speech.* IEEE SLT. arXiv:2205.12446.
- **Epitran**: Mortensen, Dalmia, Littell (2018). *Epitran: Precision G2P for Many
  Languages.* LREC.
- **PanPhon** (segmentation + PFER feature table): Mortensen, Littell, Bharadwaj,
  Goyal, Dyer, Levin (2016). *PanPhon: A Resource for Mapping IPA Segments to
  Articulatory Feature Vectors.* COLING.
