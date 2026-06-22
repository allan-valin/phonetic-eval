# MultIPA — manual run

Multilingual IPA recognizer: Wav2Vec2-Large-XLSR fine-tuned on Common Voice +
several IPA corpora (Taguchi, Sakai et al., 2023). Upstream code:
<https://github.com/ctaguchi/multipa>. Checkpoint:
<https://huggingface.co/ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns>.
Like Wav2Vec2Phoneme it is a standard HF CTC model, so you can run it straight from
`transformers`.

## 1. Install (upstream recipe)
```bash
python -m venv env_hf && source env_hf/bin/activate
pip install torch torchaudio transformers librosa soundfile
```
The repo also packages training/eval scripts:
```bash
git clone https://github.com/ctaguchi/multipa
pip install -e multipa        # optional: only needed for their CLI / eval scripts
```
> In this project use `envs/env_hf` (shared with Wav2Vec2Phoneme and WhIPA).

## 2. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).

## 3. Run — directly from `transformers`
The model is plain wav2vec2 + CTC, so the inference loop is identical to
Wav2Vec2Phoneme; only the checkpoint name changes:
```python
import torch, soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC

CKPT = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForCTC.from_pretrained(CKPT).eval()

speech, sr = sf.read("audio/myclip.wav", dtype="float32")
inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
ipa = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
print(ipa)
```
The MultIPA tokenizer's vocabulary is IPA symbols, so `batch_decode` already yields
an IPA string (space-separated phones).

## 4. Run — the authors' repo scripts (optional)
The `ctaguchi/multipa` repo provides end-to-end `train.py` / `eval.py` utilities
(dataset loading via `datasets`, their own PER/PFER with panphon). If you want to
reproduce the paper's table rather than transcribe ad-hoc files, follow the repo
README's `eval` instructions and point it at the Common Voice test split.

## 5. Notes
- MultIPA is the closest "broad-IPA" baseline to this benchmark's references and is
  the most robust model on *unseen* languages in the STIPA paper.
- Same `do_phonemize` caveat does **not** apply here — MultIPA's tokenizer is a
  plain IPA vocab and does not pull in espeak.
