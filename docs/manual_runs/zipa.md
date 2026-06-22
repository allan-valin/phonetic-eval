# ZIPA — manual run (small & large)

Efficient Zipformer-based multilingual phone recognizer (Zhu et al., ACL 2025).
Upstream: <https://github.com/lingjzhu/zipa>. ZIPA has two inference paths; the
**ONNX** path is the one to use unless you are training, because the torch path
imports `k2`/`icefall` (CUDA-bound, hard to build).

This file covers both `zipa` (small) and `zipa_large` — only the checkpoint differs.

## 1. Install (upstream ONNX recipe)
From the repo's *ONNX Inference* section:
```bash
python -m venv env_zipa && source env_zipa/bin/activate
pip install onnxruntime soundfile librosa lhotse torch
git clone https://github.com/lingjzhu/zipa        # for inference/inference.py + tokens
```
> In this project use `envs/env_zipa`, and the repo is already cloned at
> `corpora/zipa_repo/`.

> Do **not** follow the repo's `torch` training path for inference: it needs `k2`
> and `icefall` matched to your exact CUDA/torch versions. The ONNX path above has
> none of that.

## 2. Download the exported model
Each ZIPA checkpoint's HF hub has an `model.onnx` (FP32/FP16/INT8) plus a
`tokens.txt`. Grab the *Final Averaged Checkpoint*:
```bash
# small (≈64M, crctc-ns-700k)
huggingface-cli download anyspeech/zipa-small-crctc-ns-700k --local-dir models/zipa_small
# large (≈300M, crctc-ns-800k)
huggingface-cli download anyspeech/zipa-large-crctc-ns-800k --local-dir models/zipa_large
```
You need `model.onnx` and the matching `tokens.txt` from the same hub.

## 3. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).
The ONNX script extracts 80-dim Kaldi fbank features via `lhotse` internally.

## 4. Run — the authors' inference script
Single file (CTC checkpoint):
```bash
python inference/inference.py audio/myclip.wav \
    --model-path models/zipa_small/model.onnx \
    --model-type ctc \
    --tokens models/zipa_small/tokens.txt
# -> a phone sequence (greedy CTC decode)
```
A whole directory (wav/flac/mp3):
```bash
python inference/batch_inference.py audio/ \
    --model-path models/zipa_large/model.onnx \
    --model-type ctc --tokens models/zipa_large/tokens.txt --batch-size 32
```
For the **large** checkpoint just swap `--model-path` / `--tokens` to the
`models/zipa_large/` files. For FP16/INT8 variants add `--suffix .fp16.onnx` (or
`.int8.onnx`) — slightly faster, slightly worse.

## 5. Notes / gotchas
- The CTC vocab is a **127-symbol per-character IPA** inventory, so greedy-decoded
  tokens are already individual phones (space-separated).
- ZIPA emits modifiers/diacritics (`ʰ ː ˞ ̪` …) as **their own tokens**. Keep them
  as emitted; feature-based PFER re-segments them correctly. Drop only the
  sentencepiece word-boundary marker `▁` (U+2581), which is not a phone.
- `small` and `large` were near-tied at corpus scale in this benchmark — do not
  assume the larger model wins.
