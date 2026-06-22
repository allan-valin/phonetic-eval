#!/usr/bin/env python3
"""
build_fleurs_g2p.py -- build the FLEURS + G2P *contrast set*.

This is a deliberately DIFFERENT kind of test from the VoxAngeles benchmark:

  VoxAngeles            this set (FLEURS + G2P)
  --------------------  ----------------------------------------
  single words          full read-out sentences (connected speech)
  GOLD narrow IPA       SILVER broad *phonemic* IPA
  (human-audited)       (predicted by Epitran grapheme->phoneme)

So the references here are machine-generated and only as good as Epitran for the
language. We pick orthographically transparent languages (where rule-based G2P is
reliable) spanning six families. The point is contrast: connected speech + broad
predicted refs should score MUCH lower PER than narrow-gold VoxAngeles, showing how
much the benchmark's reference style drives the headline numbers. Label results as
predicted-phonemic, NOT ground truth.

Run in env_hf (needs: datasets, epitran, soundfile, panphon):
    envs/env_hf/bin/python scripts/build_fleurs_g2p.py [--utts 20]

Writes (overwriting any current test set, like the other builders):
    audio/fleurs_<iso3>_<NNN>.wav   16 kHz mono
    references.tsv                  filename, language, family, reference_ipa
"""

import argparse
import glob
import io
import os

import soundfile as sf
import panphon
from datasets import load_dataset, Audio
import epitran

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
AUDIO_DIR = os.path.join(PROJECT_DIR, "audio")
REFERENCES = os.path.join(PROJECT_DIR, "references.tsv")

# FLEURS config, Epitran code, ISO-639-3 (filename field POWSM reads), name, family.
# All chosen for transparent orthographies => reliable rule-based G2P.
LANGS = [
    ("es_419", "spa-Latn", "spa", "Spanish",    "Indo-European"),    # Romance
    ("de_de",  "deu-Latn", "deu", "German",     "Indo-European"),    # Germanic
    ("tr_tr",  "tur-Latn", "tur", "Turkish",    "Turkic"),
    ("fi_fi",  "fin-Latn", "fin", "Finnish",    "Uralic"),
    ("sw_ke",  "swa-Latn", "swa", "Swahili",    "Atlantic-Congo"),
    ("id_id",  "ind-Latn", "ind", "Indonesian", "Austronesian"),
]

MAX_SECONDS = 25.0          # skip very long clips to bound CPU inference time
ft = panphon.FeatureTable()  # for segmenting the G2P string into phones


def phone_tokenize(ipa_string):
    """Epitran output (word-spaced, phones concatenated) -> space-separated phones.

    We drop word boundaries (PER is scored over phones, like the VoxAngeles refs)
    and re-segment with panphon so each token is one IPA phone -- matching the
    reference format the scorer expects.
    """
    segs = ft.ipa_segs(ipa_string.replace(" ", ""))
    return " ".join(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", type=int, default=20, help="utterances per language")
    args = ap.parse_args()

    # Clear any previous test set so audio/ holds only this dataset.
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(AUDIO_DIR, "*.wav")):
        os.remove(old)

    rows = []
    for cfg, epi_code, iso3, name, family in LANGS:
        print(f"== {name} ({cfg}, {epi_code}) ==")
        epi = epitran.Epitran(epi_code)
        # decode=False -> raw bytes (avoids datasets' torchcodec audio decoder).
        ds = load_dataset("google/fleurs", cfg, split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))

        kept = 0
        for sample in ds:
            if kept >= args.utts:
                break
            text = (sample.get("transcription") or "").strip()
            if not text:
                continue
            data, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]),
                               dtype="float32", always_2d=True)
            data = data.mean(axis=1)                      # -> mono 1-D
            if len(data) / sr > MAX_SECONDS:              # too long: skip
                continue
            ref = phone_tokenize(epi.transliterate(text))
            if not ref:                                   # G2P produced nothing usable
                continue
            kept += 1
            fname = f"fleurs_{iso3}_{kept:03d}.wav"
            sf.write(os.path.join(AUDIO_DIR, fname), data, sr,
                     subtype="PCM_16")                    # 16-bit PCM, already 16 kHz
            rows.append((fname, name, family, ref))
        print(f"   kept {kept} utterances")

    with open(REFERENCES, "w", encoding="utf-8") as f:
        f.write("filename\tlanguage\tfamily\treference_ipa\n")
        for fname, name, family, ref in rows:
            f.write(f"{fname}\t{name}\t{family}\t{ref}\n")

    print(f"\nWrote {len(rows)} files to audio/ and references.tsv "
          f"across {len(LANGS)} langs / "
          f"{len({r[2] for r in rows})} families")


if __name__ == "__main__":
    main()
