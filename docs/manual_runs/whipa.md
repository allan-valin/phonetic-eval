# WhIPA / LoWhIPA — manual run (base-cv & base-comb)

Whisper fine-tuned for narrow-IPA output — the STIPA paper's own models (Suchardt,
El-Shazli & Cassotti, EMNLP 2025). Upstream: <https://github.com/jshrdt/whipa>.
WhIPA is **not** a drop-in HF pipeline: you load it through the repo's `WHIPA`
class, which installs the custom `<|ip|>` "IPA language" token and the fine-tuned
tokenizer.

Covers `whipa` = **base-cv** (full fine-tune) and `whipa_comb` = **base-comb**
(LoRA adapter). Both use `openai/whisper-base` as the backbone here.

## 1. Install (upstream recipe)
```bash
python -m venv env_hf && source env_hf/bin/activate
pip install torch torchaudio transformers peft panphon pandas pyyaml soundfile
git clone https://github.com/jshrdt/whipa
```
`peft` is needed for the LoRA (base-comb) variant; `panphon`/`pandas`/`pyyaml` are
imported by the repo's tokenizer + metrics utilities.
> In this project use `envs/env_hf`, and the repo is already cloned at
> `corpora/whipa_repo/` (loader class in `code/deploy.py`).

## 2. Download the fine-tuned weights
From the `jshrdt/lowhipa-models` collection on the Hub:
```bash
# base-cv: full fine-tune, trained on Common Voice (ja/pl/mt/hu/fi/el/ta)
huggingface-cli download jshrdt/whipa-base-cv     --local-dir models/whipa_base_cv
# base-comb: LoRA adapter, adds Arabic + Mandarin
huggingface-cli download jshrdt/lowhipa-base-comb --local-dir models/lowhipa_base_comb
```
The base Whisper weights (`openai/whisper-base`) are fetched automatically by
`transformers` on first load.

## 3. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).
The repo wraps audio in a `datasets.Dataset` with keys `['audio', 'input_features']`
(see `code/scripts/whipa_utils.prep_dataset`); `sample['audio']['array']` is used to
compute clip length for the phone-rate fallback.

## 4. Run — the authors' loader (`code/deploy.py`)
```python
import sys; sys.path.insert(0, "whipa/code")        # so `deploy` is importable
from deploy import WHIPA

# base-cv (full fine-tune):
whipa = WHIPA(model_path="models/whipa_base_cv",
              base_model_name="openai/whisper-base")
# base-comb (LoRA): add lora=True
# whipa = WHIPA(model_path="models/lowhipa_base_comb",
#               base_model_name="openai/whisper-base", lora=True)

# The repo's documented call (see its README "3) Inference"):
ipa_prediction = whipa.transcribe_ipa(sample)       # sample = dataset row w/ input_features
print(ipa_prediction)
```
`transcribe_ipa()` accepts decoding knobs worth tuning per the README:
`n_beams`, `fallback_beams`, `max_phones_per_sec_rate`, `repetition_penalty`,
`exponential_decay_length_penalty`.

## 5. Known bug in `transcribe_ipa()`
In the cloned revision, `transcribe_ipa()` references an **undefined global**
`whipa` and uses `torch` **without importing it**, so calling it as-is can raise
`NameError`. Two options:
- Call `whipa.transcribe_ipa(sample)` as a method (the `self` binding supplies the
  model), and ensure `import torch` is in scope; or
- Use `WHIPA` only for **loading** (it correctly sets up the `<|ip|>` token +
  tokenizer) and run the decode yourself:
  ```python
  import torch
  feats = whipa.processor(speech, sampling_rate=16000,
                          return_tensors="pt").input_features
  with torch.no_grad():
      ids = whipa.model.generate(feats, num_beams=5)[0]
  ipa = whipa.tokenizer.decode(ids, skip_special_tokens=True).strip()
  ```
  This benchmark uses the second approach plus a phone-rate fallback ladder (beam
  backoff → repetition penalty → length-decay penalty → truncate) that only fires
  when phones/second exceeds ~20 (it guards against Whisper hallucinating/looping on
  out-of-domain audio).

## 6. Notes
- The **base** backbone (74M) is for CPU speed; the paper's headline numbers use
  **whisper-large-v2** (`jshrdt/whipa-large-cv`, `jshrdt/lowhipa-large-comb`) — same
  code, just `base_model_name="openai/whisper-large-v2"`, but ~1.5B params and slow
  on CPU.
- These base variants were **not** trained on most of this benchmark's test
  languages, so the numbers here are honest zero-shot.
- The repo also ships `STIPA_METRICS` (`code/scripts/metrics.py`) — the authors'
  own PER/PFER. This project scores with its own `evaluate.py` instead, for a
  single consistent metric across all nine models.
