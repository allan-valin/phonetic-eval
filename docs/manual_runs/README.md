# Manual runs — each model the upstream way

These notes show how to run every model **the way its own authors document it**,
without this repo's `scripts/run_inference.py` wrapper. Imagine you had never seen
the wrapper: you install each project from its own README / model card, prepare the
audio yourself, run the author's command, and read the IPA off stdout or an output
file. That is exactly what each file here walks through.

One file per model (9 configs):

| File | Config(s) | Upstream project |
|---|---|---|
| [allosaurus.md](allosaurus.md) | `allosaurus` | [xinjli/allosaurus](https://github.com/xinjli/allosaurus) |
| [wav2vec2phoneme.md](wav2vec2phoneme.md) | `wav2vec2phoneme` | [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) |
| [multipa.md](multipa.md) | `multipa` | [ctaguchi/multipa](https://github.com/ctaguchi/multipa) |
| [allophant.md](allophant.md) | `allophant` | [kgnlp/allophant](https://huggingface.co/kgnlp/allophant) |
| [zipa.md](zipa.md) | `zipa` (small), `zipa_large` | [lingjzhu/zipa](https://github.com/lingjzhu/zipa) |
| [powsm.md](powsm.md) | `powsm` | [espnet/powsm](https://huggingface.co/espnet/powsm) |
| [whipa.md](whipa.md) | `whipa` (base-cv), `whipa_comb` (base-comb) | [jshrdt/whipa](https://github.com/jshrdt/whipa) |

> These are **standalone** instructions. The project's automated path (one isolated
> environment per model, driven by `run_inference.py`) is documented separately in
> the top-level [README §7–§9](../../README.md#7-setup--automatic-vs-manual). Where
> this project already built an environment that satisfies a model's upstream
> dependencies, each file points it out — but the commands below are the authors'.

---

## 0. Prepare the audio (shared by every model)

Every model here expects **16 kHz, mono, PCM WAV**. Do the conversion once and
point each model at the same folder.

### From an arbitrary audio file
```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_s16le audio/myclip.wav
```
`-ar 16000` resamples to 16 kHz, `-ac 1` downmixes to mono, `-c:a pcm_s16le` writes
uncompressed 16-bit PCM. Repeat for each clip (or loop over a directory).

### From the VoxAngeles corpus (what this benchmark uses)
VoxAngeles ships per-language word recordings plus an **audited narrow-IPA** table.
1. Download the corpus (tarball) from <https://github.com/pacscilab/voxangeles>.
2. The audited transcriptions are in `transcriptions/voxangeles_transcriptions.tsv`
   (use the `updated` column — it is the hand-corrected narrow IPA).
3. Extract the per-word clips you want, convert each to 16 kHz mono with the
   `ffmpeg` line above, and keep a parallel list of `(filename, reference_ipa)`.

### Reference transcriptions (for scoring, not inference)
Inference only needs the WAVs. To *score* a model you also need a `references.tsv`
with columns `filename`, `language`, `family`, `reference_ipa` — see the top-level
[README §10](../../README.md#10-scoring-per--pfer). Scoring itself is upstream-
independent: it compares each model's IPA string to the reference with PER/PFER.

---

## A note on environments
Each model's dependencies conflict with the others' (different `torch`,
`transformers`, `numpy` pins; ESPnet needs `pyworld`; ZIPA's torch path needs
`k2`/`icefall`). That is *why* the authors each ship their own install recipe and
why this project keeps one virtual environment per model. Never install two of
these into the same environment.
