#!/usr/bin/env python3
"""
run_inference.py  -- run one phonetic-transcription model over the test audio.

Design: each model's library is imported lazily inside its own function, so you
run this script in the model's own virtual environment and only that model's
dependencies need to be present. Output: results/<model>/<filename>.txt, one IPA
string per audio file.

Run from ~/phonetic_eval/scripts/ , one model at a time:

    source ~/phonetic_eval/envs/env_allosaurus/bin/activate
    python run_inference.py --model allosaurus
    deactivate

    source ~/phonetic_eval/envs/env_hf/bin/activate
    python run_inference.py --model wav2vec2phoneme
    python run_inference.py --model multipa
    python run_inference.py --model whipa
    deactivate

    source ~/phonetic_eval/envs/env_allophant/bin/activate
    python run_inference.py --model allophant
    deactivate

Paths assume the layout in the guide. Adjust AUDIO_DIR / RESULTS_DIR if needed.
"""

import argparse
import glob
import os

# Resolve paths relative to this script's location (scripts/ -> project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
AUDIO_DIR = os.path.join(PROJECT_DIR, "audio")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# --- Checkpoints: VERIFY these against each repo's README before running ------
W2V2P_CHECKPOINT = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MULTIPA_CHECKPOINT = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
# WhIPA (Tier 3): Whisper fine-tuned for IPA. NOT a drop-in HF pipeline — uses the
# jshrdt/whipa repo's WHIPA loader class. All variants use whisper-base for CPU
# speed. Each variant: (model dir under models/, lora?, results subfolder).
#   base-cv  : full fine-tune, trained on Common Voice (ja/pl/mt/hu/fi/el/ta only)
#              -> zero-shot on our 6 test languages.
#   base-comb: LoRA adapter, trained on Arabic (ASC) + MultIPA-CV + Mandarin
#              (THCHS-30) -> more broadly multilingual; the fairer representative.
# Each variant: (model dir under models/, lora?, results subfolder, base Whisper).
#   base-* use whisper-base (74M, CPU-friendly); large-cv is the paper's headline
#   whisper-large-v2 backbone (~1.5B) -> much slower on CPU, but ~halves base PER.
WHIPA_VARIANTS = {
    "base-cv":   ("whipa_base_cv",     False, "whipa",       "openai/whisper-base"),
    "base-comb": ("lowhipa_base_comb", True,  "whipa_comb",  "openai/whisper-base"),
    "large-cv":  ("whipa_large_cv",    False, "whipa_large", "openai/whisper-large-v2"),
}
WHIPA_REPO_CODE = os.path.join(PROJECT_DIR, "corpora", "whipa_repo", "code")
# ZIPA (Tier 3): ONNX CTC path — no k2/icefall needed. model.onnx + tokens.txt
# downloaded from an anyspeech HF hub into models/zipa_*/ (see RESUME.md).
# Each variant maps to (model dir under models/, results subfolder name).
ZIPA_VARIANTS = {
    "small": ("zipa_small_ns", "zipa"),         # 64M  crctc-ns-700k  (default)
    "large": ("zipa_large_ns", "zipa_large"),   # 300M crctc-ns-800k
}
# -----------------------------------------------------------------------------


def get_audio_files():
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    if not files:
        raise SystemExit(f"No .wav files found in {AUDIO_DIR}/")
    return files


def ensure_outdir(model_name):
    outdir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save(outdir, audio_path, ipa_string):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(outdir, base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write((ipa_string or "").strip() + "\n")
    preview = (ipa_string or "").strip()[:60]
    print(f"  {base}: {preview}")


def _load_audio_mono16k(audio_path):
    """Load audio, resample to 16kHz, collapse to mono. Returns 1-D numpy array.

    Uses soundfile for I/O (torchaudio.load now requires the optional torchcodec
    package, which is not installed). torchaudio.functional.resample does not need
    torchcodec, so it is still used if a file is not already 16 kHz.
    """
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # (frames, ch)
    data = data.mean(axis=1)                                          # -> mono 1-D
    if sr != 16000:
        import torch
        import torchaudio
        wav = torch.from_numpy(data).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        data = wav.squeeze(0).numpy()
    return data


# --- Allosaurus ---------------------------------------------------------------
def run_allosaurus():
    from allosaurus.app import read_recognizer
    outdir = ensure_outdir("allosaurus")
    model = read_recognizer()                # default universal model
    for audio in get_audio_files():
        # No language restriction => most "universal" decoding.
        ipa = model.recognize(audio)         # returns space-separated phones
        save(outdir, audio, ipa)


# --- Wav2Vec2Phoneme and MultIPA (both wav2vec2 + CTC) ------------------------
def run_hf_ctc(model_name, checkpoint):
    import torch
    from transformers import AutoProcessor, AutoModelForCTC
    outdir = ensure_outdir(model_name)
    # Wav2Vec2Phoneme's tokenizer eagerly initializes a phonemizer (espeak)
    # backend in __init__ when do_phonemize=True -- but that backend is only used
    # for text->phoneme encoding, which we never do (we only CTC-decode ids->IPA).
    # Loading with do_phonemize=False skips it and removes the espeak dependency.
    processor = AutoProcessor.from_pretrained(checkpoint, do_phonemize=False)
    model = AutoModelForCTC.from_pretrained(checkpoint)
    model.eval()
    for audio in get_audio_files():
        speech = _load_audio_mono16k(audio)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        ipa = processor.batch_decode(pred_ids)[0]
        save(outdir, audio, ipa)


# --- WhIPA (Whisper fine-tuned for IPA) ---------------------------------------
# Loads via the jshrdt/whipa repo's WHIPA class (handles the custom <|ip|> "IPA
# language" token + the fine-tuned tokenizer). The repo's transcribe_ipa() has two
# bugs (uses an undefined global `whipa` and `torch` without importing it), so we
# reimplement its decode + phone-rate fallback here, fixed, and decode with
# skip_special_tokens so the phone-count signal and saved output are clean IPA.
# The fallback exists because Whisper models can hallucinate/loop on out-of-domain
# audio; it backs off beam size, then adds a repetition penalty, then an
# exponential length penalty, until the phone/second rate is plausible.
def run_whipa(variant="base-cv"):
    import sys
    import torch
    if WHIPA_REPO_CODE not in sys.path:
        sys.path.insert(0, WHIPA_REPO_CODE)
    from deploy import WHIPA
    from scripts.metrics import retokenize_ipa

    model_subdir, lora, out_name, base_model = WHIPA_VARIANTS[variant]
    model_path = os.path.join(PROJECT_DIR, "models", model_subdir)
    outdir = ensure_outdir(out_name)
    whipa = WHIPA(model_path=model_path, base_model_name=base_model, lora=lora)
    model, tokenizer = whipa.model, whipa.tokenizer
    n_beams = whipa.ft_config.get("gen_args", {}).get("num_beams", 5)
    max_phones_per_sec = 20.0

    def decode(**gen_kwargs):
        with torch.no_grad():
            ids = model.generate(input_features, **gen_kwargs)[0]
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    def n_phones(text):
        return len(retokenize_ipa(text))

    for audio in get_audio_files():
        speech = _load_audio_mono16k(audio)
        input_features = whipa.processor(
            speech, sampling_rate=16000, return_tensors="pt").input_features
        seconds = max(round(len(speech) / 16000), 1)
        rate_limit = max(seconds * max_phones_per_sec, 1)

        out = decode(num_beams=n_beams)
        # Fallback ladder, only if the prediction overshoots the phone-rate limit.
        if n_phones(out) > rate_limit:
            backoff = [b for b in (1, 3, 5, 7) if b != n_beams]
            for beams in backoff:                       # 1) beam backoff
                if n_phones(out) <= rate_limit:
                    break
                out = decode(num_beams=beams)
            if n_phones(out) > rate_limit:              # 2) + repetition penalty
                for beams in [n_beams] + backoff:
                    if n_phones(out) <= rate_limit:
                        break
                    out = decode(num_beams=beams, repetition_penalty=1.15)
            penalty = 2.0                               # 3) + length-decay penalty
            while n_phones(out) > rate_limit and penalty <= 5:
                out = decode(num_beams=n_beams,
                             exponential_decay_length_penalty=(int(rate_limit * 0.8),
                                                               penalty))
                penalty += 1.5
            if n_phones(out) > rate_limit:              # 4) last resort: truncate
                out = " ".join(retokenize_ipa(out)[:int(rate_limit)])
        save(outdir, audio, out)


# --- Allophant (official API: Estimator + Batch + inventory) ------------------
# Allophant predicts articulatory features and composes them into phonemes from a
# given inventory. For a language-INDEPENDENT setting we decode every utterance
# against one fixed, typologically broad multilingual inventory (union below),
# applied uniformly -- we do not feed the model the true language of each file.
PHONEME_LAYER = "phoneme"
ALLOPHANT_INVENTORY_LANGS = [
    "en", "es", "tr", "cmn", "ar", "hi", "ru", "de", "fr", "ja", "ko", "vi",
    "fi", "sw", "yue", "th",
]


def run_allophant():
    import torch
    import torchaudio
    import soundfile as sf
    from allophant.estimator import Estimator
    from allophant.dataset_processing import Batch
    from allophant import predictions

    outdir = ensure_outdir("allophant")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # "kgnlp/allophant" is the feature-composition model (see model card).
    model, attribute_indexer = Estimator.restore("kgnlp/allophant", device=device)

    # Broad fixed inventory (union over diverse languages). phoneme_inventory()
    # accepts a list and returns the merged inventory.
    inventory = attribute_indexer.phoneme_inventory(ALLOPHANT_INVENTORY_LANGS)
    feature_matrix = attribute_indexer.composition_feature_matrix(inventory).to(device)
    inventory_indexer = attribute_indexer.attributes.subset(inventory)
    ctc_decoders = predictions.feature_decoders(
        inventory_indexer, feature_names=[PHONEME_LAYER])

    for audio in get_audio_files():
        data, sr = sf.read(audio, dtype="float32", always_2d=True)   # (frames, ch)
        waveform = torch.from_numpy(data.T)[:1]                      # (1, frames)
        if sr != model.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, model.sample_rate)
        batch = Batch(waveform,
                      torch.tensor([waveform.shape[1]]),
                      torch.zeros(1))
        model_outputs = model.predict(batch.to(device), feature_matrix)
        decoder = ctc_decoders[PHONEME_LAYER]
        decoded = decoder(
            model_outputs.outputs[PHONEME_LAYER].transpose(1, 0).contiguous(),
            model_outputs.lengths)
        # decoded: per-utterance list of beam candidates; take the top beam.
        hypothesis = decoded[0][0]
        phones = inventory_indexer.feature_values(PHONEME_LAYER,
                                                  hypothesis.tokens - 1)
        save(outdir, audio, " ".join(phones))


# --- ZIPA (Zipformer phone recognizer, ONNX CTC inference) --------------------
# Run in env_zipa. Uses the optimized ONNX checkpoint so neither k2 nor icefall
# is required (the torch path in the ZIPA repo needs both; the ONNX path does
# not). Acoustic features are 80-dim kaldi fbank via lhotse, exactly matching the
# ZIPA repo's inference/utils.py. The CTC vocab is a 127-symbol per-character IPA
# inventory: greedy-decoded tokens are already individual phones, so output is
# space-separated like the other models. We drop the sentencepiece word-boundary
# marker U+2581 ("LOWER ONE EIGHTH BLOCK") since it is not a phone. NOTE: ZIPA
# emits modifiers/diacritics (ʰ ː ˞ ̪ ...) as their own tokens; we keep them as
# emitted (the feature-based PFER re-segments them correctly in evaluate.py).
ZIPA_WORD_BOUNDARY = "▁"


def run_zipa(variant="small"):
    import numpy as np
    import onnxruntime as ort
    from lhotse.features.kaldi.extractors import Fbank, FbankConfig
    import torch

    model_subdir, out_name = ZIPA_VARIANTS[variant]
    zipa_dir = os.path.join(PROJECT_DIR, "models", model_subdir)
    onnx_path = os.path.join(zipa_dir, "model.onnx")
    tokens_path = os.path.join(zipa_dir, "tokens.txt")
    outdir = ensure_outdir(out_name)

    # Token id -> symbol map (tokens.txt: "<symbol> <id>" per line).
    vocab = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab[int(parts[1])] = parts[0]

    session = ort.InferenceSession(onnx_path)
    extractor = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
    blank_id = 0

    for audio in get_audio_files():
        speech = _load_audio_mono16k(audio)                  # 1-D float32, 16 kHz
        wav = torch.from_numpy(speech).float().unsqueeze(0)  # (1, samples)
        feats = extractor.extract_batch([wav], sampling_rate=16000)
        feature = feats[0].unsqueeze(0)                      # (1, T, 80)
        feat_lens = np.array([feature.shape[1]], dtype=np.int64)
        log_probs = session.run(None, {"x": feature.numpy(),
                                       "x_lens": feat_lens})[0][0]   # (T, V)
        # CTC greedy: collapse repeats, drop blanks.
        preds = np.argmax(log_probs, axis=-1)
        phones, prev = [], -1
        for idx in preds:
            if idx != blank_id and idx != prev:
                sym = vocab.get(int(idx), "")
                if sym and sym != ZIPA_WORD_BOUNDARY:
                    phones.append(sym)
            prev = idx
        save(outdir, audio, " ".join(phones))


# --- POWSM (OWSM/Whisper-style multitask speech model, ESPnet) ----------------
# Run in env_powsm. POWSM is an encoder-decoder S2T model; task "<pr>" performs
# phone recognition, emitting IPA phones each wrapped in slashes, e.g.
# "<isl><pr><notimestamps> /p//e//iː//θ/". We strip the symbol prefix at
# <notimestamps> and recover phones by splitting on "/" (keeps phone boundaries
# as space-separated tokens, unlike the model card's replace("/","") which would
# concatenate them). POWSM conditions on a language symbol per utterance; we feed
# the file's ISO 639-3 code (the middle field of the filename) when POWSM knows
# it, else its built-in "<unk>". NOTE: of our six languages only isl/yue/mya are
# in POWSM's inventory; pes/lug/cpn fall back to <unk> (record this asymmetry).
def run_powsm():
    from espnet2.bin.s2t_inference import Speech2Text

    outdir = ensure_outdir("powsm")
    s2t = Speech2Text.from_pretrained(
        "espnet/powsm", device="cpu", task_sym="<pr>")
    valid_syms = set(s2t.converter.token_list)

    for audio in get_audio_files():
        iso = os.path.splitext(os.path.basename(audio))[0].split("_")[1]
        sym = f"<{iso}>"
        if sym not in valid_syms:
            sym = "<unk>"
        speech = _load_audio_mono16k(audio)
        pred = s2t(speech, text_prev="<na>", lang_sym=sym, task_sym="<pr>")[0][0]
        if "<notimestamps>" in pred:
            pred = pred.split("<notimestamps>")[-1]
        phones = [tok for tok in pred.split("/") if tok.strip()]
        save(outdir, audio, " ".join(phones))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["allosaurus", "wav2vec2phoneme", "multipa", "whipa",
                 "allophant", "zipa", "powsm"],
    )
    parser.add_argument(
        "--variant", default=None,
        help="Model variant. zipa: small (default) | large. "
             "whipa: base-cv (default) | base-comb.",
    )
    args = parser.parse_args()
    print(f"Running {args.model} -> results/{args.model}/")

    if args.model == "allosaurus":
        run_allosaurus()
    elif args.model == "wav2vec2phoneme":
        run_hf_ctc("wav2vec2phoneme", W2V2P_CHECKPOINT)
    elif args.model == "multipa":
        run_hf_ctc("multipa", MULTIPA_CHECKPOINT)
    elif args.model == "whipa":
        run_whipa(args.variant or "base-cv")
    elif args.model == "allophant":
        run_allophant()
    elif args.model == "zipa":
        run_zipa(args.variant or "small")
    elif args.model == "powsm":
        run_powsm()
    print("Done.")


if __name__ == "__main__":
    main()
