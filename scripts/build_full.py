#!/usr/bin/env python3
"""
build_full.py -- build audio/ + references.tsv for the ENTIRE VoxAngeles set.

Same extraction/segmentation logic as build_testset.py, but with NO balancing
quotas: it takes every utterance in every language that (a) has audited IPA, (b)
segments to >=2 phones, and (c) has audio in its language zip. Filenames follow
the same <family>_<lang>_<NNN>.wav scheme so evaluate.py can still group by
family. Rewrites audio/ + references.tsv from scratch.

    source ~/phonetic_eval/envs/env_score/bin/activate   # panphon
    FFMPEG=envs/tools/bin/ffmpeg python scripts/build_full.py
"""
import csv, os, shutil, subprocess, sys, zipfile

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


def family_token(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_meta():
    fam_name = {}
    with open(FAMILIES_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                fam_name[row[0].strip()] = row[1].strip()
    meta = {}
    with open(GLOTTO_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = (row.get("iso_6393") or "").strip()
            if not iso:
                continue
            meta[iso] = {
                "name": (row.get("Language") or row.get("name") or iso).strip(),
                "family": fam_name.get((row.get("family_id") or "").strip(), "Unknown"),
            }
    return meta


def main():
    if not os.path.exists(FFMPEG):
        sys.exit(f"ffmpeg not found at {FFMPEG}; set $FFMPEG")
    try:
        import panphon
    except ImportError:
        sys.exit("panphon not importable -- run inside env_score")
    ft = panphon.FeatureTable()
    meta = load_meta()

    # group usable utterances per language
    by_lang = {}
    with open(TRANS_TSV, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lang = (r.get("lang") or "").strip()
            fid = (r.get("file") or "").strip()
            updated = (r.get("updated") or "").strip()
            if not (lang and fid and updated and lang in meta):
                continue
            seg = " ".join(ft.ipa_segs(updated))
            if len(seg.split()) < 2:
                continue
            by_lang.setdefault(lang, []).append((fid, seg))

    if os.path.isdir(AUDIO_DIR):
        for fn in os.listdir(AUDIO_DIR):
            if fn.endswith(".wav"):
                os.remove(os.path.join(AUDIO_DIR, fn))
    os.makedirs(AUDIO_DIR, exist_ok=True)

    ref_rows = []
    for lang in sorted(by_lang):
        zpath = os.path.join(ALIGNED_DIR, lang + ".zip")
        if not os.path.exists(zpath):
            continue
        with zipfile.ZipFile(zpath) as z:
            members = {os.path.splitext(os.path.basename(n))[0]: n
                       for n in z.namelist() if n.lower().endswith(".wav")}
            fam = meta[lang]["family"]
            prefix = family_token(fam)
            utts = [(fid, seg) for fid, seg in sorted(by_lang[lang]) if fid in members]
            for i, (fid, seg) in enumerate(utts, 1):
                out_name = f"{prefix}_{lang}_{i:03d}.wav"
                out_path = os.path.join(AUDIO_DIR, out_name)
                tmp_raw = os.path.join(AUDIO_DIR, f".raw_{out_name}")
                with z.open(members[fid]) as src, open(tmp_raw, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", tmp_raw,
                                "-ar", "16000", "-ac", "1", out_path], check=True)
                os.remove(tmp_raw)
                ref_rows.append({"filename": out_name, "language": meta[lang]["name"],
                                 "family": fam, "reference_ipa": seg})
        print(f"  {meta[lang]['family']:18s} {lang}: {len(utts)} utts")

    with open(REFERENCES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, delimiter="\t",
                           fieldnames=["filename", "language", "family", "reference_ipa"])
        w.writeheader()
        w.writerows(ref_rows)
    fams = sorted({r["family"] for r in ref_rows})
    print(f"\nWrote {len(ref_rows)} files to audio/ and references.tsv across "
          f"{len({r['filename'].split('_')[1] for r in ref_rows})} langs / {len(fams)} families")


if __name__ == "__main__":
    main()
