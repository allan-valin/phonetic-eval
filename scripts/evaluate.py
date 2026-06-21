#!/usr/bin/env python3
"""
evaluate.py -- compute PER and PFER for every model against the references.

Reads:
    references.tsv          (filename, language, family, reference_ipa)
    results/<model>/*.txt   (one IPA hypothesis per audio file)

Writes:
    per_file_results.tsv    every file x model, with PER and PFER
    summary_by_model.tsv    mean PER + PFER per model
    summary_by_family.tsv   mean PER + PFER per model per language family

Run in an environment with:  pip install jiwer panphon
From the project root:        python scripts/evaluate.py

Metric notes:
- PER: edit distance over phones, normalized by reference length. jiwer treats
  whitespace-separated tokens as units, so phones must be space-separated.
- PFER: panphon feature edit distance (substitution cost scales with the number
  of differing articulatory features; insert/delete cost 1), normalized by the
  number of reference phones. Matches the spirit of Taguchi & Sakai (2023).

Cleaning notes (see guide section 9): model outputs are normalized to canonical
Unicode and stripped of special tokens before scoring. Extend clean_ipa() as you
discover model-specific quirks.
"""

import csv
import os
import re
import statistics
import unicodedata

import jiwer
import panphon.distance

RESULTS_DIR = "results"
REFERENCES = "references.tsv"

dst = panphon.distance.Distance()

# Look-alike code points that should be unified before scoring.
LOOKALIKE_MAP = {
    "g": "\u0261",   # LATIN SMALL LETTER G -> IPA SCRIPT G
    ":": "\u02d0",   # colon -> IPA length mark
    "'": "\u02c8",   # apostrophe -> primary stress
}

# Tokens some models emit that are not phones.
SPECIAL_TOKEN_RE = re.compile(r"<\|?[^>|]*\|?>")   # <|en|>, <pad>, etc.


def clean_ipa(text):
    """Normalize a raw model output (or reference) to canonical IPA for scoring."""
    if not text:
        return ""
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(LOOKALIKE_MAP.get(ch, ch) for ch in text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_phone_tokens(text):
    """Return a space-separated phone string for PER (jiwer needs delimiters)."""
    cleaned = clean_ipa(text)
    if " " in cleaned:
        return cleaned                      # already phone-separated (Allosaurus)
    # No spaces: segment into phones with panphon so PER is meaningful.
    segs = dst.fm.ipa_segs(cleaned.replace(" ", ""))
    return " ".join(segs) if segs else cleaned


def load_references():
    refs = {}
    with open(REFERENCES, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = os.path.splitext(row["filename"])[0]
            refs[key] = {
                "language": row["language"],
                "family": row["family"],
                "reference_ipa": row["reference_ipa"].strip(),
            }
    return refs


def discover_models():
    return sorted(
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )


def load_hypothesis(model, key):
    path = os.path.join(RESULTS_DIR, model, key + ".txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def compute_per(reference, hypothesis):
    ref_tokens = to_phone_tokens(reference)
    hyp_tokens = to_phone_tokens(hypothesis)
    if not ref_tokens:
        return None
    return jiwer.wer(ref_tokens, hyp_tokens)


def compute_pfer(reference, hypothesis):
    ref = clean_ipa(reference).replace(" ", "")
    hyp = clean_ipa(hypothesis).replace(" ", "")
    if not ref:
        return None
    fed = dst.feature_edit_distance(ref, hyp)
    n_phones = len(dst.fm.ipa_segs(ref)) or 1
    return fed / n_phones


def main():
    refs = load_references()
    models = discover_models()
    if not models:
        raise SystemExit(f"No model folders found in {RESULTS_DIR}/")
    print(f"Models found: {', '.join(models)}")

    per_file_rows = []
    results = {m: [] for m in models}               # model -> [(per, pfer), ...]
    family_results = {}                              # (model, family) -> [...]

    for key, ref in refs.items():
        for model in models:
            hyp = load_hypothesis(model, key)
            if hyp is None:
                print(f"  WARNING: no output for {model}/{key}")
                continue
            per = compute_per(ref["reference_ipa"], hyp)
            pfer = compute_pfer(ref["reference_ipa"], hyp)
            per_file_rows.append({
                "filename": key, "language": ref["language"],
                "family": ref["family"], "model": model,
                "PER": f"{per:.4f}" if per is not None else "",
                "PFER": f"{pfer:.4f}" if pfer is not None else "",
            })
            if per is not None and pfer is not None:
                results[model].append((per, pfer))
                family_results.setdefault((model, ref["family"]), []).append((per, pfer))

    with open("per_file_results.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, delimiter="\t",
                           fieldnames=["filename", "language", "family",
                                       "model", "PER", "PFER"])
        w.writeheader()
        w.writerows(per_file_rows)

    with open("summary_by_model.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "n", "mean_PER", "mean_PFER"])
        for model in models:
            vals = results[model]
            if vals:
                w.writerow([model, len(vals),
                            f"{statistics.mean(v[0] for v in vals):.4f}",
                            f"{statistics.mean(v[1] for v in vals):.4f}"])
            else:
                w.writerow([model, 0, "", ""])

    with open("summary_by_family.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "family", "n", "mean_PER", "mean_PFER"])
        for (model, family), vals in sorted(family_results.items()):
            w.writerow([model, family, len(vals),
                        f"{statistics.mean(v[0] for v in vals):.4f}",
                        f"{statistics.mean(v[1] for v in vals):.4f}"])

    print("\nWrote: per_file_results.tsv, summary_by_model.tsv, summary_by_family.tsv")
    print("\nQuick summary by model:")
    for model in models:
        vals = results[model]
        if vals:
            print(f"  {model:18s} PER={statistics.mean(v[0] for v in vals):.3f} "
                  f"PFER={statistics.mean(v[1] for v in vals):.3f} (n={len(vals)})")


if __name__ == "__main__":
    main()
