# Phonetic Eval — Previous Test Results (160-file balanced set)

_Snapshot generated 2026-06-21 from the existing scored run (RESUME.md dated 2026-06-02)._
_Test set: 8 families x 2 langs x 10 utts = 160 files. Metrics: PER (exact-match phone error rate), PFER (panphon feature error rate). Lower is better._

Use this file to **manually re-verify**: re-run the command for any model, re-score, and check the numbers below still match.

## Overall (mean over 160 files), sorted best PER first

| model | n | mean PER | mean PFER | reproduce |
|---|---|---|---|---|
| powsm | 160 | 0.5165 | 0.1380 | `envs/env_powsm/bin/python scripts/run_inference.py --model powsm` |
| wav2vec2phoneme | 160 | 0.5297 | 0.1728 | `envs/env_hf/bin/python scripts/run_inference.py --model wav2vec2phoneme` |
| multipa | 160 | 0.5939 | 0.1439 | `envs/env_hf/bin/python scripts/run_inference.py --model multipa` |
| zipa | 160 | 0.6772 | 0.1696 | `envs/env_zipa/bin/python scripts/run_inference.py --model zipa --variant small` |
| zipa_large | 160 | 0.6842 | 0.1744 | `envs/env_zipa/bin/python scripts/run_inference.py --model zipa --variant large` |
| allophant | 160 | 0.7067 | 0.2170 | `envs/env_allophant/bin/python scripts/run_inference.py --model allophant` |
| whipa_comb | 160 | 0.8313 | 0.2252 | `envs/env_hf/bin/python scripts/run_inference.py --model whipa --variant base-comb` |
| whipa | 160 | 0.8373 | 0.2767 | `envs/env_hf/bin/python scripts/run_inference.py --model whipa --variant base-cv` |
| allosaurus | 160 | 0.9206 | 0.3071 | `envs/env_allosaurus/bin/python scripts/run_inference.py --model allosaurus` |

Re-score after any re-run with:
```bash
cd ~/phonetic_eval && envs/env_score/bin/python scripts/evaluate.py
column -t -s$'\t' summary_by_model.tsv
```

## Spot-check prediction (file: `indoeuropean_isl_001.wav`)

Reference IPA: `b iː ð`

| model | predicted phones |
|---|---|
| allosaurus | `b̥ i ð` |
| wav2vec2phoneme | `b iː θ` |
| multipa | `piθ` |
| allophant | `b iɪ d̪` |
| zipa | `p i t s` |
| zipa_large | `b i θ` |
| powsm | `p e iː θ` |
| whipa | `biz` |
| whipa_comb | `piːtː` |

## Per-family breakdown (mean PER / mean PFER, n=20 each)

### allophant

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.7175 | 0.2540 |
| Atlantic-Congo | 20 | 0.7050 | 0.2422 |
| Austronesian | 20 | 0.5792 | 0.1754 |
| Dravidian | 20 | 0.6825 | 0.2168 |
| Indo-European | 20 | 0.7417 | 0.1575 |
| Nakh-Daghestanian | 20 | 0.8119 | 0.3116 |
| Sino-Tibetan | 20 | 0.7700 | 0.2227 |
| Uralic | 20 | 0.6458 | 0.1555 |

### allosaurus

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.9767 | 0.3482 |
| Atlantic-Congo | 20 | 0.9917 | 0.3769 |
| Austronesian | 20 | 0.7861 | 0.2061 |
| Dravidian | 20 | 0.9717 | 0.3162 |
| Indo-European | 20 | 0.9125 | 0.2454 |
| Nakh-Daghestanian | 20 | 0.9661 | 0.3512 |
| Sino-Tibetan | 20 | 0.7650 | 0.2273 |
| Uralic | 20 | 0.9950 | 0.3855 |

### multipa

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.5300 | 0.1707 |
| Atlantic-Congo | 20 | 0.4867 | 0.0791 |
| Austronesian | 20 | 0.5354 | 0.1220 |
| Dravidian | 20 | 0.5058 | 0.1240 |
| Indo-European | 20 | 0.8750 | 0.1495 |
| Nakh-Daghestanian | 20 | 0.7437 | 0.2319 |
| Sino-Tibetan | 20 | 0.6783 | 0.1743 |
| Uralic | 20 | 0.3967 | 0.0992 |

### powsm

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.5567 | 0.1501 |
| Atlantic-Congo | 20 | 0.4483 | 0.1485 |
| Austronesian | 20 | 0.4320 | 0.1161 |
| Dravidian | 20 | 0.4767 | 0.1072 |
| Indo-European | 20 | 0.8000 | 0.1858 |
| Nakh-Daghestanian | 20 | 0.6756 | 0.1938 |
| Sino-Tibetan | 20 | 0.4542 | 0.1070 |
| Uralic | 20 | 0.2883 | 0.0956 |

### wav2vec2phoneme

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.5150 | 0.1625 |
| Atlantic-Congo | 20 | 0.3392 | 0.1318 |
| Austronesian | 20 | 0.3594 | 0.1144 |
| Dravidian | 20 | 0.5000 | 0.1445 |
| Indo-European | 20 | 0.7125 | 0.1912 |
| Nakh-Daghestanian | 20 | 0.6240 | 0.2171 |
| Sino-Tibetan | 20 | 0.7242 | 0.2043 |
| Uralic | 20 | 0.4633 | 0.2163 |

### whipa

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.6625 | 0.2112 |
| Atlantic-Congo | 20 | 0.8400 | 0.2015 |
| Austronesian | 20 | 0.5100 | 0.1312 |
| Dravidian | 20 | 0.6867 | 0.1733 |
| Indo-European | 20 | 0.9375 | 0.2398 |
| Nakh-Daghestanian | 20 | 0.9319 | 0.3254 |
| Sino-Tibetan | 20 | 0.8283 | 0.2115 |
| Uralic | 20 | 1.3017 | 0.7196 |

### whipa_comb

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.7075 | 0.2144 |
| Atlantic-Congo | 20 | 0.7675 | 0.1553 |
| Austronesian | 20 | 0.6794 | 0.2011 |
| Dravidian | 20 | 0.7550 | 0.2187 |
| Indo-European | 20 | 1.0125 | 0.2535 |
| Nakh-Daghestanian | 20 | 1.0099 | 0.3370 |
| Sino-Tibetan | 20 | 0.8383 | 0.1888 |
| Uralic | 20 | 0.8800 | 0.2325 |

### zipa

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.5917 | 0.1529 |
| Atlantic-Congo | 20 | 0.4992 | 0.1884 |
| Austronesian | 20 | 0.5512 | 0.1326 |
| Dravidian | 20 | 0.6850 | 0.1338 |
| Indo-European | 20 | 0.9958 | 0.0982 |
| Nakh-Daghestanian | 20 | 0.7815 | 0.2364 |
| Sino-Tibetan | 20 | 0.7767 | 0.2893 |
| Uralic | 20 | 0.5367 | 0.1251 |

### zipa_large

| family | n | mean PER | mean PFER |
|---|---|---|---|
| Afro-Asiatic | 20 | 0.6000 | 0.1462 |
| Atlantic-Congo | 20 | 0.5067 | 0.1989 |
| Austronesian | 20 | 0.5399 | 0.1109 |
| Dravidian | 20 | 0.7975 | 0.1850 |
| Indo-European | 20 | 0.8625 | 0.0933 |
| Nakh-Daghestanian | 20 | 0.8369 | 0.3081 |
| Sino-Tibetan | 20 | 0.7850 | 0.2503 |
| Uralic | 20 | 0.5450 | 0.1024 |
