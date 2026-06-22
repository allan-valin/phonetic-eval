# Wav2Vec2Phoneme — manual run

Wav2Vec2-XLSR-53 fine-tuned for multilingual phoneme recognition with an espeak
phoneme tokenizer (Xu et al., 2021). Upstream model card:
<https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft>. It is a standard
Hugging Face CTC model — no custom repo, just `transformers`.

## 1. Install (upstream recipe)
```bash
python -m venv env_hf && source env_hf/bin/activate
pip install torch torchaudio transformers librosa soundfile
```
> In this project use `envs/env_hf` (it already has torch + transformers +
> torchaudio + librosa + soundfile, and is shared with MultIPA and WhIPA).

## 2. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).
The processor expects a raw float waveform sampled at 16 kHz.

## 3. Run — the model card's snippet
```python
import torch, soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC

CKPT = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForCTC.from_pretrained(CKPT).eval()

speech, sr = sf.read("audio/myclip.wav", dtype="float32")   # mono 16 kHz
inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred_ids = torch.argmax(logits, dim=-1)
ipa = processor.batch_decode(pred_ids)[0]                   # space-separated phones
print(ipa)
```

## 4. The one real gotcha: espeak
The processor's tokenizer can eagerly initialise an **espeak** phonemizer backend
in its constructor. That backend is only used to convert *text → phonemes*, which
you never do for inference (you only CTC-decode ids → IPA). Two ways around it:

- **Install espeak** so the tokenizer loads normally:
  ```bash
  sudo apt-get install espeak-ng        # or: pip install phonemizer + espeak-ng
  ```
- **Skip it** — load the processor with phonemization disabled (no espeak needed):
  ```python
  processor = AutoProcessor.from_pretrained(CKPT, do_phonemize=False)
  ```
  This benchmark uses the second option (no system package required).

## 5. Notes
- Output is space-separated phonemes, ready for PER scoring.
- This is a **broad phonemic** model (espeak-style inventory), so expect the
  broad-vs-narrow gap against the audited narrow-IPA references.
