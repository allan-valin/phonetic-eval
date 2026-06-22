# Allophant — manual run

Multilingual phone recognizer that predicts **articulatory features** and composes
them into phonemes from a supplied inventory (Glocker et al., 2023). Upstream
package + model card: <https://huggingface.co/kgnlp/allophant>, code:
<https://github.com/kgnlp/allophant>.

## 1. Install (upstream recipe)
```bash
python -m venv env_allophant && source env_allophant/bin/activate
pip install allophant
```
This pulls torch, torchaudio and the `allophant` package. Add `soundfile` for audio
I/O if it is not already present.
> In this project the prebuilt environment is `envs/env_allophant` (made with `uv`;
> it has **no** `pip` — use `uv pip install ...` or `python -m ...` inside it).

## 2. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).
Allophant resamples internally to `model.sample_rate`, but feeding 16 kHz mono
avoids surprises.

## 3a. Run — the authors' CLI (simplest)
The package installs an `allophant` command:
```bash
allophant predict -m kgnlp/allophant audio/myclip.wav
```
> Heads-up: on some dependency combinations the CLI clashes (a `numcodecs`/`zarr`
> version conflict). If it errors on import, use the Python API below, which does
> not go through the CLI entry point.

## 3b. Run — the authors' Python API (model card)
Allophant is *language-independent* when you decode against a fixed, broad phoneme
inventory rather than the file's true language. The model-card path:
```python
import torch, torchaudio, soundfile as sf
from allophant.estimator import Estimator
from allophant.dataset_processing import Batch
from allophant import predictions

PHONEME = "phoneme"
# A broad multilingual inventory (union over typologically diverse languages):
LANGS = ["en","es","tr","cmn","ar","hi","ru","de","fr","ja","ko","vi","fi","sw","yue","th"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, indexer = Estimator.restore("kgnlp/allophant", device=device)

inventory   = indexer.phoneme_inventory(LANGS)
feat_matrix = indexer.composition_feature_matrix(inventory).to(device)
inv_indexer = indexer.attributes.subset(inventory)
decoders    = predictions.feature_decoders(inv_indexer, feature_names=[PHONEME])

data, sr = sf.read("audio/myclip.wav", dtype="float32", always_2d=True)
wave = torch.from_numpy(data.T)[:1]                      # (1, frames)
if sr != model.sample_rate:
    wave = torchaudio.functional.resample(wave, sr, model.sample_rate)

batch = Batch(wave, torch.tensor([wave.shape[1]]), torch.zeros(1))
out = model.predict(batch.to(device), feat_matrix)
hyp = decoders[PHONEME](out.outputs[PHONEME].transpose(1, 0).contiguous(),
                        out.lengths)[0][0]               # top beam
phones = inv_indexer.feature_values(PHONEME, hyp.tokens - 1)
print(" ".join(phones))
```
Feed a single fixed inventory to every file for the language-agnostic setting; pass
a single language's inventory to `phoneme_inventory([...])` for the
language-conditioned setting.

## 4. Notes / gotchas
- The **inventory choice matters**: Allophant can only output phonemes that are in
  the inventory you give it. A narrow inventory raises precision on that language; a
  broad union is the fair language-independent comparison used here.
- Prefer the **Estimator API** over the CLI in constrained environments (see the
  `numcodecs`/`zarr` note above).
- `hyp.tokens - 1` accounts for the CTC blank offset when mapping token ids back to
  inventory symbols.
