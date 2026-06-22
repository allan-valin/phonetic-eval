# Allosaurus — manual run

Universal phone recognizer (Li et al., 2020). Upstream:
<https://github.com/xinjli/allosaurus>. It ships as a pip package with its own CLI,
so this is the simplest model to run by hand.

## 1. Install (upstream recipe)
```bash
python -m venv env_allosaurus && source env_allosaurus/bin/activate
pip install allosaurus
```
On first use it downloads its pretrained model (`uni2005`) to the package cache.
To pre-fetch or pin it explicitly:
```bash
python -m allosaurus.bin.download_model -m uni2005
python -m allosaurus.bin.list_model        # see what's installed
```
> In this project the ready-made environment is `envs/env_allosaurus`
> (`allosaurus 1.0.2`, model `uni2005` already cached).

## 2. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).
Allosaurus is tolerant of sample rate but 16 kHz mono is what the model expects.

## 3. Run — the authors' CLI
Single file, universal (language-independent) decoding:
```bash
python -m allosaurus.run -i audio/myclip.wav
# -> a space-separated IPA phone string on stdout, e.g.:  p e i θ
```
Useful flags from the upstream README:
- `--lang <id>` restricts the phone inventory to one language (e.g. `--lang eng`).
  Omit it (or use the default) for the most universal output — that is what this
  benchmark uses, so no language is fed to the model.
- `--model <name>` selects a checkpoint (default `uni2005`).
- `--topk <k>` emits the k best phones per frame.

## 4. Run — the authors' Python API
For a folder of files, the documented API is more convenient than the CLI:
```python
from allosaurus.app import read_recognizer

model = read_recognizer()                 # default universal model (uni2005)
ipa = model.recognize("audio/myclip.wav") # -> "p e i θ"  (space-separated phones)
print(ipa)
```
Loop over your `audio/*.wav` and write each `ipa` string to
`results/allosaurus/<clip>.txt` if you want to score it with this repo's
`evaluate.py`.

## 5. Notes / gotchas
- Output is **already space-separated phones**, so no extra segmentation is needed
  for PER.
- Decoding with no `--lang` gives the broadest inventory; restricting the language
  generally lowers error on that language but is not the language-agnostic setting
  this benchmark measures.
- Allosaurus emits **broad** phones (few diacritics), so against narrow audited IPA
  it shows the broad-vs-narrow PER gap discussed in the top-level README §4.
