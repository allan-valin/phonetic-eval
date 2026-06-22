# POWSM — manual run

Phone-aware OWSM: an ESPnet S2T (OWSM/Whisper-style) multitask speech model whose
`<pr>` task performs phone recognition. Upstream model card:
<https://huggingface.co/espnet/powsm> (runs through ESPnet).

## 1. Install (upstream recipe — mind `pyworld`)
```bash
pip install espnet espnet_model_zoo torch torchaudio
```
**Gotcha:** ESPnet pulls in `pyworld`, which has no prebuilt wheel for many Python
versions and needs a C compiler to build from source. On a box with **no gcc**,
install via conda/micromamba instead, which provides a prebuilt `pyworld` plus
compilers:
```bash
micromamba create -n env_powsm python=3.10 -c conda-forge pyworld
micromamba activate env_powsm
pip install espnet espnet_model_zoo torch torchaudio
```
> In this project the prebuilt environment is `envs/env_powsm` (a **micromamba**
> env, py3.10, conda-forge `pyworld`). Activate it with
> `micromamba activate ~/phonetic_eval/envs/env_powsm` or call its python directly.

## 2. Prepare audio
16 kHz mono WAV (see [README §0](README.md#0-prepare-the-audio-shared-by-every-model)).
Load it to a float array (e.g. with `soundfile`) before calling the model.

## 3. Run — the authors' Speech2Text API
```python
import soundfile as sf
from espnet2.bin.s2t_inference import Speech2Text

s2t = Speech2Text.from_pretrained("espnet/powsm", device="cpu", task_sym="<pr>")
valid_syms = set(s2t.converter.token_list)

speech, sr = sf.read("audio/myclip.wav", dtype="float32")   # mono 16 kHz
lang_sym = "<isl>"                       # ISO-639-3 of the utterance, if known
if lang_sym not in valid_syms:
    lang_sym = "<unk>"                   # fall back when POWSM doesn't know the language

pred = s2t(speech, text_prev="<na>", lang_sym=lang_sym, task_sym="<pr>")[0][0]
# pred looks like: "<isl><pr><notimestamps> /p//e//iː//θ/"
if "<notimestamps>" in pred:
    pred = pred.split("<notimestamps>")[-1]
phones = [tok for tok in pred.split("/") if tok.strip()]
print(" ".join(phones))                  # -> "p e iː θ"
```

## 4. Output parsing — do NOT use `replace("/","")`
POWSM wraps each phone in slashes (`/p//e//iː/`). The model card's
`replace("/", "")` concatenates them into one unsegmented blob (`peiː`), which
destroys phone boundaries and inflates PER. **Split on `/`** instead (as above) to
recover space-separated phones.

## 5. Language symbol caveat
POWSM conditions on a per-utterance language symbol. Its inventory covers ~92
languages; many low-resource ISO codes are absent and fall back to `<unk>`. In this
benchmark that fallback did **not** obviously hurt accuracy (an `<unk>` language was
sometimes POWSM's *best*), but record which utterances used a real symbol vs.
`<unk>` when you write up results — it is a confound, not a controlled variable.

## 6. Notes
- Slowest model on CPU here (~2 s/clip); budget accordingly for large sets.
- Encoder-decoder (not CTC), so it can hallucinate on out-of-domain audio — sanity
  check a few outputs.
