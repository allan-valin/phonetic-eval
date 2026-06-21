#!/usr/bin/env python3
"""
build_testset.py -- build audio/ + references.tsv from the VoxAngeles corpus.

VoxAngeles ships audited, canonical IPA per utterance (the `updated` column of
transcriptions/voxangeles_transcriptions.tsv) plus per-language audio inside
data/audited_aligned/<lang>.zip. This script:

  1. maps every VoxAngeles language (ISO 639-3) to its language family using the
     repo's own glottolog tables (lrec-coling_analyses/map/),
  2. selects a balanced subset (N families x K languages x M utterances),
  3. extracts each chosen WAV from its language zip and resamples it to 16 kHz
     mono with ffmpeg into audio/<family>_<lang>_<NNN>.wav,
  4. segments the audited IPA into phones with panphon and writes references.tsv
     (filename, language, family, reference_ipa) with space-separated phones.

Run in an env that has panphon (env_score) and with ffmpeg on PATH (tools env):

    source ~/phonetic_eval/envs/env_score/bin/activate
    python build_testset.py --families 3 --langs-per-family 2 --utts 8   # pilot

The family prefix in the filename lets evaluate.py group results by family.
Re-run with larger numbers later to expand the set; it rewrites audio/ +
references.tsv from scratch each time.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VOX_DIR = os.path.join(PROJECT_DIR, "corpora", "voxangeles")
TRANS_TSV = os.path.join(VOX_DIR, "transcriptions", "voxangeles_transcriptions.tsv")
MAP_DIR = os.path.join(VOX_DIR, "lrec-coling_analyses", "map")
GLOTTO_CSV = os.path.join(MAP_DIR, "ucla_glottlog.csv")
FAMILIES_CSV = os.path.join(MAP_DIR, "lang_families.csv")
ALIGNED_DIR = os.path.join(VOX_DIR, "data", "audited_aligned")
AUDIO_DIR = os.path.join(PROJECT_DIR, "audio")
REFERENCES = os.path.join(PROJECT_DIR, "references.tsv")

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or
                        os.path.join(PROJECT_DIR, "envs", "tools", "bin", "ffmpeg"))

# Priority order for family selection: phonologically diverse, well-known
# families known to be present in VoxAngeles. Families not listed here are still
# eligible (appended in alphabetical order) if more are requested than available.
FAMILY_PRIORITY = [
    "Indo-European", "Atlantic-Congo", "Sino-Tibetan", "Afro-Asiatic",
    "Austronesian", "Turkic", "Dravidian", "Uralic", "Nakh-Daghestanian",
    "Austroasiatic", "Mande", "Abkhaz-Adyge", "Siouan", "Salishan",
]


def family_token(name):
    """Filename-safe lowercase token for a family name (e.g. Indo-European -> indoeuropean)."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_lang_metadata():
    """iso_6393 -> {"name": human language name, "family": family display name}."""
    fam_name = {}
    with open(FAMILIES_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                fam_name[row[0].strip()] = row[1].strip()

    meta = {}
    with open(GLOTTO_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = (row.get("iso_6393") or "").strip()
            fam_id = (row.get("family_id") or "").strip()
            if not iso:
                continue
            meta[iso] = {
                "name": (row.get("Language") or row.get("name") or iso).strip(),
                "family": fam_name.get(fam_id, fam_id or "Unknown"),
            }
    return meta


def load_transcriptions():
    """Return list of (lang, file_id, updated_ipa) for rows with non-empty audited IPA."""
    rows = []
    with open(TRANS_TSV, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            updated = (r.get("updated") or "").strip()
            if r.get("lang") and r.get("file") and updated:
                rows.append((r["lang"].strip(), r["file"].strip(), updated))
    return rows


def segment_ipa(text, ft):
    """panphon segmentation -> space-separated phones; '' if nothing usable."""
    segs = ft.ipa_segs(text)
    return " ".join(segs)


def zip_members(lang):
    """Set of available wav basenames (file ids) inside a language zip."""
    path = os.path.join(ALIGNED_DIR, lang + ".zip")
    if not os.path.exists(path):
        return None, None
    with zipfile.ZipFile(path) as z:
        names = {os.path.splitext(os.path.basename(n))[0]: n
                 for n in z.namelist() if n.lower().endswith(".wav")}
    return path, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=3)
    ap.add_argument("--langs-per-family", type=int, default=2)
    ap.add_argument("--utts", type=int, default=8, help="utterances per language")
    ap.add_argument("--seed", type=int, default=0, help="(reserved) deterministic; selection is sorted")
    args = ap.parse_args()

    if not os.path.exists(FFMPEG):
        sys.exit(f"ffmpeg not found at {FFMPEG}; set $FFMPEG or put it on PATH")
    try:
        import panphon
    except ImportError:
        sys.exit("panphon not importable -- run inside env_score")
    ft = panphon.FeatureTable()

    meta = load_lang_metadata()
    rows = load_transcriptions()

    # Group usable utterances by family -> lang -> [(file_id, seg_ipa)]
    by_family = {}
    for lang, file_id, updated in rows:
        m = meta.get(lang)
        if not m:
            continue
        seg = segment_ipa(updated, ft)
        if len(seg.split()) < 2:        # need a couple of real phones
            continue
        by_family.setdefault(m["family"], {}).setdefault(lang, []).append((file_id, seg))

    # Order families: priority list first, then the rest alphabetically.
    avail = set(by_family)
    ordered_families = [f for f in FAMILY_PRIORITY if f in avail]
    ordered_families += sorted(f for f in avail if f not in FAMILY_PRIORITY)

    selected = []   # (family, lang)
    for fam in ordered_families:
        if len([s for s in selected]) and len({f for f, _ in selected}) >= args.families \
           and fam not in {f for f, _ in selected}:
            break
        if len({f for f, _ in selected}) >= args.families:
            break
        langs = by_family[fam]
        # langs with enough utterances, most-data first then alphabetical
        eligible = sorted((l for l in langs if len(langs[l]) >= args.utts),
                          key=lambda l: (-len(langs[l]), l))
        if len(eligible) < args.langs_per_family:
            continue                    # skip families that can't fill the quota
        for lang in eligible[:args.langs_per_family]:
            selected.append((fam, lang))

    chosen_fams = {f for f, _ in selected}
    if len(chosen_fams) < args.families:
        print(f"WARNING: only {len(chosen_fams)} families could be filled "
              f"(requested {args.families}). Lower --utts/--langs-per-family for more.")

    # Fresh output
    if os.path.isdir(AUDIO_DIR):
        for fn in os.listdir(AUDIO_DIR):
            if fn.endswith(".wav"):
                os.remove(os.path.join(AUDIO_DIR, fn))
    os.makedirs(AUDIO_DIR, exist_ok=True)

    ref_rows = []
    for fam, lang in selected:
        zpath, members = zip_members(lang)
        if not members:
            print(f"  skip {fam}/{lang}: no audio zip")
            continue
        utts = [(fid, seg) for fid, seg in sorted(by_family[fam][lang])
                if fid in members][:args.utts]
        prefix = family_token(fam)
        with zipfile.ZipFile(zpath) as z:
            for i, (fid, seg) in enumerate(utts, 1):
                out_name = f"{prefix}_{lang}_{i:03d}.wav"
                out_path = os.path.join(AUDIO_DIR, out_name)
                tmp_raw = os.path.join(AUDIO_DIR, f".raw_{out_name}")
                with z.open(members[fid]) as src, open(tmp_raw, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", tmp_raw,
                                "-ar", "16000", "-ac", "1", out_path], check=True)
                os.remove(tmp_raw)
                ref_rows.append({
                    "filename": out_name,
                    "language": meta[lang]["name"],
                    "family": fam,
                    "reference_ipa": seg,
                })
        print(f"  {fam:16s} {lang}: {len(utts)} utts")

    with open(REFERENCES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, delimiter="\t",
                           fieldnames=["filename", "language", "family", "reference_ipa"])
        w.writeheader()
        w.writerows(ref_rows)

    fam_summary = sorted({(r["family"], r["language"]) for r in ref_rows})
    print(f"\nWrote {len(ref_rows)} files to audio/ and references.tsv")
    print(f"Families: {sorted({r['family'] for r in ref_rows})}")
    print(f"Languages: {[l for _, l in fam_summary]}")


if __name__ == "__main__":
    main()
