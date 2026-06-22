# Phonetic Eval — Previous Test Results (160-file balanced set)

_Snapshot regenerated 2026-06-22 from the scored 160-file run (`runs/160/`), now including the diacritic-folded metric._
_Test set: 8 families x 2 langs x 10 utts = 160 files. Metrics: PER (exact-match phone error rate), PFER (panphon feature error rate). Lower is better._

**Two granularities** (see top-level [README §10](../README.md#10-scoring-per--pfer)): _strict_ scores the raw narrow IPA; _folded_ (`PER_f` / `PFER_f`) scores after `fold_to_broad()` strips diacritics, length and tone, collapsing narrow refs (`iː`, `b̥`, `t̪`) to broad phones (`i`, `b`, `t`). The strict→folded PER drop quantifies how much error is pure reference granularity.

Use this file to **manually re-verify**: re-run the command for any model, re-score, and check the numbers below still match.

## Overall (mean over 160 files), sorted best PER first

| model | n | mean PER | mean PFER | PER_f | PFER_f | reproduce |
|---|---|---|---|---|---|---|
| powsm | 160 | 0.5165 | 0.1380 | 0.4210 | 0.1347 | `envs/env_powsm/bin/python scripts/run_inference.py --model powsm` |
| wav2vec2phoneme | 160 | 0.5297 | 0.1728 | 0.3874 | 0.1541 | `envs/env_hf/bin/python scripts/run_inference.py --model wav2vec2phoneme` |
| multipa | 160 | 0.5939 | 0.1439 | 0.4767 | 0.1411 | `envs/env_hf/bin/python scripts/run_inference.py --model multipa` |
| zipa | 160 | 0.6772 | 0.1696 | 0.5203 | 0.1453 | `envs/env_zipa/bin/python scripts/run_inference.py --model zipa --variant small` |
| zipa_large | 160 | 0.6842 | 0.1744 | 0.5269 | 0.1502 | `envs/env_zipa/bin/python scripts/run_inference.py --model zipa --variant large` |
| allophant | 160 | 0.7067 | 0.2170 | 0.4631 | 0.2015 | `envs/env_allophant/bin/python scripts/run_inference.py --model allophant` |
| whipa_comb | 160 | 0.8313 | 0.2252 | 0.5777 | 0.2025 | `envs/env_hf/bin/python scripts/run_inference.py --model whipa --variant base-comb` |
| whipa | 160 | 0.8373 | 0.2767 | 0.6893 | 0.2680 | `envs/env_hf/bin/python scripts/run_inference.py --model whipa --variant base-cv` |
| allosaurus | 160 | 0.9206 | 0.3071 | 0.8245 | 0.3143 | `envs/env_allosaurus/bin/python scripts/run_inference.py --model allosaurus` |

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

## Per-family breakdown (mean PER / mean PFER / folded PER_f / folded PFER_f, n=20 each)

### allophant

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.7175 | 0.2540 | 0.5392 | 0.2477 |
| Atlantic-Congo | 20 | 0.7050 | 0.2422 | 0.5050 | 0.2380 |
| Austronesian | 20 | 0.5792 | 0.1754 | 0.2931 | 0.1655 |
| Dravidian | 20 | 0.6825 | 0.2168 | 0.5325 | 0.2078 |
| Indo-European | 20 | 0.7417 | 0.1575 | 0.4458 | 0.1263 |
| Nakh-Daghestanian | 20 | 0.8119 | 0.3116 | 0.4932 | 0.2923 |
| Sino-Tibetan | 20 | 0.7700 | 0.2227 | 0.5217 | 0.2175 |
| Uralic | 20 | 0.6458 | 0.1555 | 0.3742 | 0.1172 |

### allosaurus

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.9767 | 0.3482 | 0.8258 | 0.3397 |
| Atlantic-Congo | 20 | 0.9917 | 0.3769 | 0.9333 | 0.3955 |
| Austronesian | 20 | 0.7861 | 0.2061 | 0.6973 | 0.2374 |
| Dravidian | 20 | 0.9717 | 0.3162 | 0.8467 | 0.3506 |
| Indo-European | 20 | 0.9125 | 0.2454 | 0.7292 | 0.2435 |
| Nakh-Daghestanian | 20 | 0.9661 | 0.3512 | 0.9019 | 0.3243 |
| Sino-Tibetan | 20 | 0.7650 | 0.2273 | 0.7133 | 0.2172 |
| Uralic | 20 | 0.9950 | 0.3855 | 0.9483 | 0.4063 |

### multipa

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.5300 | 0.1707 | 0.4192 | 0.1645 |
| Atlantic-Congo | 20 | 0.4867 | 0.0791 | 0.3825 | 0.0731 |
| Austronesian | 20 | 0.5354 | 0.1220 | 0.3601 | 0.1100 |
| Dravidian | 20 | 0.5058 | 0.1240 | 0.3267 | 0.1071 |
| Indo-European | 20 | 0.8750 | 0.1495 | 0.7917 | 0.1315 |
| Nakh-Daghestanian | 20 | 0.7437 | 0.2319 | 0.6277 | 0.2380 |
| Sino-Tibetan | 20 | 0.6783 | 0.1743 | 0.5558 | 0.1919 |
| Uralic | 20 | 0.3967 | 0.0992 | 0.3500 | 0.1128 |

### powsm

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.5567 | 0.1501 | 0.4025 | 0.1601 |
| Atlantic-Congo | 20 | 0.4483 | 0.1485 | 0.4067 | 0.1452 |
| Austronesian | 20 | 0.4320 | 0.1161 | 0.3514 | 0.1082 |
| Dravidian | 20 | 0.4767 | 0.1072 | 0.3683 | 0.0976 |
| Indo-European | 20 | 0.8000 | 0.1858 | 0.6167 | 0.1743 |
| Nakh-Daghestanian | 20 | 0.6756 | 0.1938 | 0.6114 | 0.1869 |
| Sino-Tibetan | 20 | 0.4542 | 0.1070 | 0.4025 | 0.1142 |
| Uralic | 20 | 0.2883 | 0.0956 | 0.2083 | 0.0912 |

### wav2vec2phoneme

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.5150 | 0.1625 | 0.3650 | 0.1546 |
| Atlantic-Congo | 20 | 0.3392 | 0.1318 | 0.2975 | 0.1294 |
| Austronesian | 20 | 0.3594 | 0.1144 | 0.2864 | 0.1090 |
| Dravidian | 20 | 0.5000 | 0.1445 | 0.3708 | 0.1313 |
| Indo-European | 20 | 0.7125 | 0.1912 | 0.4833 | 0.1684 |
| Nakh-Daghestanian | 20 | 0.6240 | 0.2171 | 0.4527 | 0.2041 |
| Sino-Tibetan | 20 | 0.7242 | 0.2043 | 0.5242 | 0.1544 |
| Uralic | 20 | 0.4633 | 0.2163 | 0.3192 | 0.1816 |

### whipa

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.6625 | 0.2112 | 0.5192 | 0.2122 |
| Atlantic-Congo | 20 | 0.8400 | 0.2015 | 0.5900 | 0.1963 |
| Austronesian | 20 | 0.5100 | 0.1312 | 0.3794 | 0.1202 |
| Dravidian | 20 | 0.6867 | 0.1733 | 0.4783 | 0.1606 |
| Indo-European | 20 | 0.9375 | 0.2398 | 0.8792 | 0.2261 |
| Nakh-Daghestanian | 20 | 0.9319 | 0.3254 | 0.8067 | 0.3326 |
| Sino-Tibetan | 20 | 0.8283 | 0.2115 | 0.6333 | 0.2042 |
| Uralic | 20 | 1.3017 | 0.7196 | 1.2283 | 0.6923 |

### whipa_comb

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.7075 | 0.2144 | 0.4492 | 0.2078 |
| Atlantic-Congo | 20 | 0.7675 | 0.1553 | 0.4308 | 0.1361 |
| Austronesian | 20 | 0.6794 | 0.2011 | 0.4868 | 0.1858 |
| Dravidian | 20 | 0.7550 | 0.2187 | 0.4892 | 0.1875 |
| Indo-European | 20 | 1.0125 | 0.2535 | 0.7667 | 0.2396 |
| Nakh-Daghestanian | 20 | 1.0099 | 0.3370 | 0.7150 | 0.2941 |
| Sino-Tibetan | 20 | 0.8383 | 0.1888 | 0.5642 | 0.1520 |
| Uralic | 20 | 0.8800 | 0.2325 | 0.7200 | 0.2175 |

### zipa

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.5917 | 0.1529 | 0.4150 | 0.1460 |
| Atlantic-Congo | 20 | 0.4992 | 0.1884 | 0.3742 | 0.1842 |
| Austronesian | 20 | 0.5512 | 0.1326 | 0.4119 | 0.1268 |
| Dravidian | 20 | 0.6850 | 0.1338 | 0.5017 | 0.1217 |
| Indo-European | 20 | 0.9958 | 0.0982 | 0.6292 | 0.0712 |
| Nakh-Daghestanian | 20 | 0.7815 | 0.2364 | 0.6907 | 0.2156 |
| Sino-Tibetan | 20 | 0.7767 | 0.2893 | 0.6333 | 0.2031 |
| Uralic | 20 | 0.5367 | 0.1251 | 0.5067 | 0.0938 |

### zipa_large

| family | n | mean PER | mean PFER | PER_f | PFER_f |
|---|---|---|---|---|---|
| Afro-Asiatic | 20 | 0.6000 | 0.1462 | 0.4033 | 0.1389 |
| Atlantic-Congo | 20 | 0.5067 | 0.1989 | 0.4025 | 0.1946 |
| Austronesian | 20 | 0.5399 | 0.1109 | 0.3906 | 0.1057 |
| Dravidian | 20 | 0.7975 | 0.1850 | 0.5475 | 0.1723 |
| Indo-European | 20 | 0.8625 | 0.0933 | 0.5958 | 0.0671 |
| Nakh-Daghestanian | 20 | 0.8369 | 0.3081 | 0.7556 | 0.2902 |
| Sino-Tibetan | 20 | 0.7850 | 0.2503 | 0.6150 | 0.1631 |
| Uralic | 20 | 0.5450 | 0.1024 | 0.5050 | 0.0696 |
