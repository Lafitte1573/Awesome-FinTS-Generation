# Verification Report — `Awesome-FinTS-Generation` README rewrite

This file is the audit trail for the metadata work. It records **how** each record was
confirmed and, more importantly, **what was wrong before**. Keep it with the repository so
future corrections can be checked rather than re-derived.

Date of verification: 2026-10-04
Scope: every entry in the previous `README.md` (48 entries, 47 unique after dedup), plus
26 works cited in the survey that the repository did not yet list, plus 26 related surveys,
plus 8 general-purpose forecasting comparators — **115 entries** in the current README.

The current README is organised by the survey's **two-level taxonomy** (task × technique), taken
from Figure 1 of `ijcai26.tex:443-486`. See §8.

---

## 1. Verification method

| Source | Used for | Notes |
|---|---|---|
| Crossref REST API | primary source for published records | matched by title similarity ≥ 0.90, then re-checked author list by hand |
| `arxiv.org/abs/<id>` (direct fetch) | preprints, and any record Crossref cannot resolve | arXiv DOIs (`10.48550/…`) are registered with DataCite, **not** Crossref |
| DBLP | preferred, but unavailable | the API is behind an Anubis anti-bot challenge for automated clients; verified entries against Crossref/arXiv instead |
| Web search | locating records Crossref missed | used only to *find* a record, never to supply metadata directly |

Every link in the new README is a `https://doi.org/…` or a `https://arxiv.org/abs/…` URL.
There are **no dead links**; the previous README had 34 `[[NICK]]()` wiki-style links whose
targets were empty.

---

## 2. Errors found and corrected

### 2.1 Wrong method name, title and year

| Nickname | Problem | Correct record |
|---|---|---|
| `LLM4FTS` | The method is **GPT4FTS**, not LLM4FTS. Title and year were both wrong (2024). | *Beyond Fixed Patches: Enhancing GPTs for Financial Prediction with Adaptive Segmentation and Learnable Wavelets*, Renjun Jia, Zian Liu, Peng Zhu, Dawei Cheng, Yuqi Liang — arXiv:2505.02880 (2025) |

### 2.2 Nicknames attached to the wrong paper

The old README used a **famous method's nickname** as the label for an unrelated paper. In
each case the paper itself is real; only the label was wrong.

| Old nickname | Actual paper | Fix |
|---|---|---|
| `Chronos` | Valeyre & Aboura, *Large Language Models for Time Series: an Application for Single Stocks and Statistical Arbitrage* | Renamed `Chronos-Arbitrage`; a genuine `Chronos` entry (Ansari et al., arXiv:2403.07815, TMLR 2024) added alongside |
| `Time-LLM` | A Noguer i Alonso & Franklin benchmark paper | Renamed `LLM-TS-Benchmark`; a genuine `Time-LLM` entry (Jin et al., ICLR 2024) added alongside |
| `SGP-LSTM` (×2) | Two genuinely different papers sharing the first author Qi Li | Split into `SGP-LSTM` (Scientific Reports 2024) and `SGP-LSTM-CX` |

### 2.3 Truncated or corrupted titles

| Old title | Correct title |
|---|---|
| `A Novel Hybrid Deep Learning` | *A Novel Hybrid Deep Learning Method for Accurate Exchange Rate Prediction* (Risks, 2024) |
| `A Novel Hybrid Deep Learning Method for Accura` | *(duplicate of the above — merged)* |
| `Robust Synthetic Data Generation for Sequentia` | *Robust Synthetic Data Generation for Sequential Financial Models Using Hybrid VAE-GRU-MCMC* (Future Internet, 2025) |
| `Article Enhancing Financial Time Series Prediction with Quantum-Enhanced…` | *Enhancing Financial Time Series Prediction with Quantum-Enhanced Synthetic Data Generation…* |
| `Article Enhancing Portfolio Performance through…` | *Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation* |
| `MARS: A FINANCIAL MARKET SIMULATION ENGINE…` (all caps) | *MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model* |
| `FINMEM: A PERFORMANCE-ENHANCED LLM TRADING AGENT…` (all caps) | *FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design* |
| `LLM4FTS: Enhancing Large Language Models for Financial Time Series Prediction` | *Beyond Fixed Patches…* (see 2.1) |

### 2.4 Wrong author name

- `FinMem` first author was recorded as **"Yangyang Vu"**; the correct spelling is
  **Yangyang Yu**. Confirmed against the Crossref record for the AAAI Symposium Series
  (DOI `10.1609/aaaiss.v3i1.31290`) and arXiv:2311.13743.

### 2.5 Wrong year

| Nickname | Old | Correct | Basis |
|---|---|---|---|
| `DiGA` | 2023 | **2026** | AAAI vol. 40 proceedings record, DOI `10.1609/aaai.v40i1.37009` |
| `Tail-GAN` | 2025 | **2026** | Management Science 72(4):2917–2936, DOI `10.1287/mnsc.2023.00936` |
| `MarS` | 2024 (preprint only) | **2025** | published at ICLR 2025; arXiv:2409.07486 retained as the preprint |
| `Edge-GAN` | 2021 | **2024** | IEEE ICNC 2024, DOI `10.1109/ICNC59896.2024.10556140` |
| `MTS-Anomaly` | 2023 | **2024** | DOI `10.4018/joeuc.342094` |
| `Distress-Imputation` | 2024 | **2022** | DOI `10.1080/0952813X.2022.2153278` |
| `GAN-MoEx` | 2023 | **2024** | IEEE CiFer 2024, DOI `10.1109/CIFEr62890.2024.10772763` |
| `NVF-DPGAN` | 2024 | **2025** | DOI `10.26599/bdma.2024.9020047` (the `2024` in the DOI is the manuscript year, not the issue year) |

`FCI` is a special case: Crossref reports **2023** (online first, DOI
`10.1515/snde-2022-0115`) but the print issue is **2025**, vol. 29(1):19–38. The print year is
used, which also matches the survey's own bibliography.

### 2.6 Duplicates removed

- `MVO-BiGRU` appeared twice (both with truncated titles) → merged into one entry.
- `GraphSAGE-CTGAN` appeared twice, identically → deduplicated.
- `SGP-LSTM` was used for two different papers → split, see 2.2.

### 2.7 Entry removed entirely

- **`CT-GAN+NAR-NN`** — the title was recorded as the literal string `RESEARCH ARTICLE`, with
  no venue, no DOI and no arXiv identifier. No record matching the described study
  (CT-GAN augmentation + NAR-NN forecasting for green-finance data, Abdelhady et al.) could be
  located. Because we could not confirm it exists in the form described, it was **removed**
  rather than kept with a guessed citation. If the authors can supply the source, it can be
  reinstated.

### 2.8 Taxonomy mismatch with the paper

The old README used **"Augmentation"** as the third task and the code `FTSA`. The published
survey uses **"Synthesis"** and the code `FTSS` (the string `FTSA` survives in the paper only
as the filename of Figure 3). The old README also assigned bare `FTSE` to benchmarks that the
paper's Table 2 marks `FTSE / FTSI`. All task codes and the Datasets table have been aligned
with the paper.

---

## 3. Records that could not be fully resolved

| Entry | Status |
|---|---|
| `Swiss-Discount-Curve` | Only an SSRN preprint exists (DOI `10.2139/ssrn.4611310`); no journal version located. Also not a time-series generator, so it sits in *Adjacent*. |
| `BERT-ABSA` | Record is complete and verified (DOI `10.1007/978-3-031-62362-2_8`), but the paper is about aspect-based sentiment analysis, not time series. Recommend dropping it entirely rather than keeping it in *Adjacent*. |

No other entry in the new README is missing a link.

### Entries without an abstract

60 of the 71 entries carry an abstract. The remaining 11 are paywalled IEEE and ACM proceedings
papers whose publishers deposit no abstract with the Crossref record:

`AttnWGAIN`, `LSSDM`, `MTSCI`, `SPDM`, `N-BEATS-GAN`, `PhysicaA-GAN`, `DM-Denoiser`,
`CoMeTS-GAN`, `CTS-GAN`, `MC-TE-GAN`, `style-facts`.

For these the README shows the task and a working link rather than a summary. Writing one from
the title alone would mean inventing method details, which is exactly the failure mode this
rewrite was meant to eliminate. If you want them filled in, the abstract needs to be read from
the paper.

---

## 3a. Placement decisions

These entries were moved out of the FTSG task sections. The reason is recorded here rather than
as an inline note in the README, so the README itself stays a clean reading list.

### Moved to *Related Surveys*

| Entry | Reason |
|---|---|
| `GAN-Fin-Review` | A systematic review, not a primary method. |

### Moved to *Adjacent & Non-Time-Series Work*

| Entry | Reason |
|---|---|
| `GraphSAGE-CTGAN` | CTGAN over tabular borrower data for credit risk — no time series. Was also duplicated twice in the old README. |
| `Swiss-Discount-Curve` | Yield-curve fitting by kernel ridge regression — not a time-series generator. |
| `Distress-Imputation` | Tabular feature imputation for financial distress, not time-series imputation. |
| `MTS-Anomaly` | Uses GAN augmentation to *detect* anomalies; the generator is not the contribution. |
| `FinanceNLP-CA` | Text augmentation for extracting board biographies. |
| `FCLM` | Tabular transaction classification for anti-money laundering. |
| `BERT-ABSA` | Aspect-based sentiment analysis — no time series at all. Recommend dropping. |

---

## 4. Works added (26 new entries)

The survey cites 39 works in its FTSG sections that the repository did not list. Of those, 26
are genuinely new entries to this README, 2 were already present under wrong nicknames
(`Data Augmentation of High Frequency Financial Data Using GAN` → `WGAN-HF-Aug`, and
`Time Series Generation with GANs for Momentum Effect Simulation…` → `GAN-MoEx`), and 11 were
deliberately excluded (see below).

**Imputation — 11 new:** CWGAIN-GP, AttnWGAIN, ImputeGAN, CSDI, LSSDM, SaSDim, MTSCI, LSCD,
SPDM, Score-CDM, FCI

**Synthesis — 13 new:** N-BEATS-GAN, QuantGAN, Tail-GAN, PhysicaA-GAN, GBMDiff, DM-Denoiser,
T2S, CoMeTS-GAN, CTS-GAN, Market-GAN, MC-TE-GAN, style-facts, stylized-facts
(plus 2 renamed rather than added: `WGAN-HF-Aug`, `GAN-MoEx`)

**Extrapolation — 2 new:** `Chronos` (Ansari et al., arXiv:2403.07815) and `Time-LLM`
(Jin et al., ICLR 2024), added to resolve the two mislabelled entries described in §2.2

### Forecasting baselines — fourth section (10 new)

Added 2026-10-04 on request. The survey cites these in its `FTSE` chapter as **comparison
baselines** rather than as FTSG contributions, so they live in their own section and the three
FTSG task lists stay strictly about generation.

| Nickname | Record |
|---|---|
| `N-BEATS` | ICLR 2020, arXiv:1905.10437 — Oreshkin, Carpov, Chapados, Bengio |
| `TimeGrad` | ICML 2021, arXiv:2101.12072 — Rasul, Seward, Schuster, Vollgraf |
| `TimeDiT` | ICML 2024 Workshop on Foundation Models in the Wild, arXiv:2409.02322 — Cao, Ye, Zhang, Liu |
| `OneFitsAll` | NeurIPS 2023, DOI 10.52202/075280-1877 — Zhou, Niu, Wang, Sun, Jin |
| `CALF` | AAAI 2025, DOI 10.1609/aaai.v39i18.34082 — Liu, Guo, Dai, Li, Bao, Ren, Jiang, Xia |
| `LLM-PS` | IEEE ICDM 2025, DOI 10.1109/icdm65498.2025.00081, pp. 733–742 — Tang, Chen, Gong, Zhang, Tao |
| `FinCast` | ACM CIKM 2025, DOI 10.1145/3746252.3761261, pp. 4539–4549 — Zhu, Chen, Qu, Chung |
| `TimeHF` | arXiv:2501.15942 (2025) — Qi, Hu, Lei, Zhang, Shi, Huang, Chen, Lin, Shen |
| `BiLSTM-ARIMA` | *Algorithms* 18(8):517, 2025, DOI 10.3390/a18080517 — Qin, Ye, Li, Cai, Gao, Qi, Ding |
| `CEEMDAN-Informer-LSTM` | *Applied Soft Computing* 177:113241, 2025, DOI 10.1016/j.asoc.2025.113241 — Li, Sun, Wu, Tao |

The 11th item in the original exclusion list, `IRM` (*Large Scale Financial Time Series
Forecasting with Multi-faceted Model*), was **already present** in the repository under
`IRM-MultiFaceted` in the FTSE section, so it is cross-referenced there rather than duplicated.
This is why the section holds 10 entries.

### Note on this round

Three arXiv identifiers were guessed during drafting and all three were wrong — `2107.01229`
is a paper on Levi-flat hypersurfaces, `2409.16038` on star–planet radio interactions, and
`2502.09111` on Gaussian-splatting SLAM. The correct identifiers above were then read off the
arXiv abstract pages directly. Recorded here because it is the failure mode this repository
most needs to avoid.


---

## 5. Images

### `assets/cover.png` — new cover figure

The README hero image is now a purpose-built figure rather than the old hand-drawn diagram.
It is generated from vector source, so every label is real text and the file stays crisp at any
size.

| File | Role |
|---|---|
| `assets/cover.png` | 3720 × 2256 raster, referenced by the README hero block |
| `assets/cover.svg` | vector source — edit this, not the PNG |

The figure shows the project's actual positioning: a two-level taxonomy plate, one column per
FTSG task, with the technique families the survey reviews listed beneath each. It uses the
current task codes (`FTSE` / `FTSI` / **`FTSS`**), so it does not repeat the `FTSA` error carried
by the old artwork.

To regenerate after an edit:

```bash
python3 assets/cover.svg -- > assets/cover.png     # or any SVG→PNG renderer
```

### `assets/Survey_00.png` — legacy, no longer referenced

Kept in the repository but no longer used by the README. It is superseded by `cover.png` and
still carries the `FTSA` label described below. Delete it once you are satisfied with the new
cover.

### `assets/taxonomy.png` — **replaced by the survey's Figure 1**

The hand-drawn taxonomy plate is gone. `assets/taxonomy.png` is now an export of **Figure 1 of the
survey**, rendered from the compiled PDF, so the README taxonomy and the paper taxonomy cannot
drift apart:

| Property | Value |
|---|---|
| Source | page 2 of `ijcai26.pdf` (the `figure*` forest environment at `ijcai26.tex:399-493`) |
| Render | `pdftoppm -f 2 -l 2 -r 600 -png`, then cropped to the diagram's ink bounding box |
| Size | 3716 × 2516, 687 KB |

Because the figure is now generated from the manuscript, all four stale items that this file used
to flag as "needs hand editing" are resolved automatically — `Financial Time Series Synthesis`
(not `Augmentation`), `GPT4FTS [Jia et al., 2025]` (not `LLM4FTS`), `TAIL-GAN [Cont et al., 2026]`
and `WGAN-BiLSTM [Kang, 2024]`, plus the current `(§ 3)/(§ 4)/(§ 5)` markers.

To regenerate after a manuscript change:

```bash
pdftoppm -f 2 -l 2 -r 600 -png ijcai26.pdf assets/_fig1   # then crop to the diagram bounds
```

Note that the figure renders the paper's *exemplar* taxonomy — representative method names, not the
full corpus. The README's Literature Review carries more works per leaf than the figure lists, and
marks the leaves where it does so with *(beyond Figure 1)*.

---

## 6. Corrections made to the manuscript

The repository audit surfaced three errors that were **in the paper**, not just the repository.

| Where | Was | Now |
|---|---|---|
| `ijcai261.bib`, `ijcai26.bib` — `LLM4FTS` entry | Title *"LLM4FTS: Enhancing Large Language Models for Financial Time Series Prediction"* with 2 of 5 authors, one mangled as `R. Jia`. The `eprint` was already correct (`2505.02880`), so a correct arXiv ID was carrying a fabricated title. | *GPT4FTS: Beyond Fixed Patches: Enhancing GPTs for Financial Prediction with Adaptive Segmentation and Learnable Wavelets*, all 5 authors |
| `ijcai26.tex:456` (Figure 1) and `ijcai26.tex:791` (body) | `LLM4FTS` | `GPT4FTS` (the `\cite{LLM4FTS}` key is unchanged, so no other edit is needed) |
| `ijcai261.bib`, `ijcai26.bib` — `Tail-gan` entry | Year 2025, no volume/issue/pages/DOI, and author names rendered as `C. Mihai`, `X. Renyuan`, `Z. Chao` | 2026, 72(4):2917–2936, DOI 10.1287/mnsc.2023.00936, authors restored to `Cucuringu, Mihai` / `Xu, Renyuan` / `Zhang, Chao` |

`LLM4FTS` → `GPT4FTS` is the same character count in both places, so the paper still compiles to
7 pages of body text plus 3 of references, with 0 LaTeX errors.


The machine-readable inputs and outputs are in this directory:

| File | Contents |
|---|---|
| `zh_abstracts.json` | the 48 original entries as extracted from the old README (fields, Chinese abstracts) |
| `en_abstracts.json` | faithful English translations of those 48 abstracts |
| `fts_dataset.json` | the final verified dataset, shaped as the two-level taxonomy: `TAXONOMY` (task → family → leaf → entries), `BASELINES`, `SURVEYS`, `ADJACENT` |
| `assets/index.csv` | flat projection of the above: section, family, leaf, nick, DOI / arXiv id |

`fts_dataset.json` is the single source of truth for the README. Regenerating the README from
it keeps the two in sync.

---

## 7. Related Surveys — expansion and relevance matching

The section previously held a single review. It now holds **26**, grouped by how each one sits
against this survey's scope rather than by year or venue.

### How candidates were found

| Channel | Result |
|---|---|
| DBLP | unavailable — the site serves an Anubis anti-bot challenge to both scripted clients and a real browser session |
| `web_search` tool | unavailable — returned `MATRIX_TOOL_REQUEST_FAILED` on every attempt |
| OpenAlex REST API | used as the discovery channel; `from_publication_date` filters must be full ISO dates (`2022-01-01`, not `2022-`) |
| Crossref REST API | used to resolve and confirm every candidate |
| Google via the in-app browser | used for discovery when OpenAlex ranking missed close-overlap work; **never** used to supply metadata |
| `arxiv.org/abs/<id>` | used for preprints |

Google surfaced one candidate that OpenAlex had not returned and that is the single most important
addition — see below. Discovery by search alone was therefore not redundant.

### Candidate rejection

| Candidate | Reason |
|---|---|
| `Kasprzak-MarketRec`, DOI `10.1016/j.neucom.2020.121856` | **rejected** — identifier was written from memory; Crossref returns 404, no such record |
| *Generative AI and Large Language Models for Financial Markets* (Iklassov et al., DOI 10.1145/3770855.3816459) | verified real, KDD 2026 — but it is a **research paper**, not a review, so it does not belong in Related Surveys |
| *Interpretable GenAI: Synthetic Financial Time Series Generation with Probabilistic LSTM* (Schwarz, SSRN 4877007) | topically the closest match of all, but an unrefereed single-method preprint — moved to **Adjacent** rather than listed as a review |
| *Bootstrapping Financial Time Series* (Ruiz & Pascual, 2002) | 24 years old and about inference under non-stationarity, not generation |

### The closest overlap, and what it changes

> **New Money: A Systematic Review of Synthetic Data Generation for Finance** — James Meldrum,
> Basem Suleiman, Fethi Rabhi, Muhammad Johan Alibasa. arXiv:2510.26076, submitted 30 Oct 2025,
> 37 pages, cs.LG.

Verified from the arXiv abstract page. Its own abstract states *"As the first systematic review
dedicated to synthetic data generation for finance, this study fills a notable gap in the
literature."*

This is the review a reviewer is most likely to raise, so it is listed **first** in group S1 with
the comparison stated explicitly rather than buried. The distinction that still holds:

| | *New Money* | This survey |
|---|---|---|
| Primary framing | synthetic data as a **privacy / regulatory** remedy | generation as a **task family** |
| Tasks distinguished | not separated | `FTSE` / `FTSI` / `FTSS` |
| Organising axis | method families under a privacy motivation | task × technique, two orthogonal levels |
| Techniques | GANs and VAEs | + diffusion models, + time-series foundation models |
| Evaluation | not the organising concern | benchmark datasets and evaluation measures are a first-class contribution |

### Groups

| Group | Entries | Framing |
|---|---|---|
| **S1** Synthetic and generative data for finance | 3 | closest overlap — read these first |
| **S2** Financial time-series surveys | 10 | forecasting / explainability / non-stationarity rather than generation |
| **S3** Foundation models and LLMs for time series | 3 | the model families placed under the FTSE branch |
| **S4** Time-series generation and synthetic data (non-financial) | 7 | technique- and evaluation-level surveys |
| **S5** General forecasting surveys | 3 | the positioning anchors the paper cites |

Two entries in S2 are the closest *methodological* siblings rather than competitors:
`Cabral-Nonstationarity` (also a taxonomy-based survey, but of drift handling) and
`Arsenault-XAI-FinTS` (paradigm-scoped, and cited by the paper itself).

Every survey entry carries a **Scope vs. this survey** line stating both what it covers and where it
stops. That is the comparison the repository promises, and it is derived from each survey's own
title, venue and abstract — never from an assumed reading of its contents.

### Year convention

Where Crossref exposes both `published-print` and `published-online`, the **print** year is used,
matching the Tail-GAN decision in §2.5. This moves several records one year later than the
originally discovered value: `Zhang-DL-Price` 2023 → 2024, `Olorunnimbe-StockDL` 2022 → 2023,
`Lin-DiffTS` 2023 → 2024, `Yang-DiffTS-STM` 2025 → 2026, `Benidis-DL-TSF` 2022 → 2023.

---

## 8. Two-level taxonomy restructure

The previous Literature Review was organised by task only. It is now organised by **task ×
technique**, mirroring Figure 1 of the survey (`ijcai26.tex:443-486`).

### Structure

| Code | Task | Families | Leaves |
|---|---|---|---|
| **A** | Financial Time Series Extrapolation (FTSE) | A1 Deep Learning Models · A2 Generative Learning Models · A3 Time-series Foundation Models | A1.1–A1.3, A2.1–A2.2, A3.1–A3.3 |
| **B** | Financial Time Series Imputation (FTSI) | B1 Machine Learning Models · B2 Generative Learning Models | B1.1–B1.2, B2.1–B2.2 |
| **C** | Financial Time Series Synthesis (FTSS) | C1 Micro-level Synthesis · C2 Macro-level Simulation | C1.1–C1.4, C2.1–C2.3 |

Leaf names are the paper's own, verbatim. Heading depth is `###` task → `####` family → `#####`
leaf → `######` entry, so every node is directly linkable.

### Moves forced by the taxonomy

Ten entries were sitting in the old flat partitions in places the paper's Figure 1 does not put
them. All ten were moved into the Literature Review:

| Entry | Was | Now | Why |
|---|---|---|---|
| `TimeGrad`, `TimeDiT` | Baselines | `A2.2 Diffusion models` | the figure names both under FTSE → Generative → Diffusion |
| `Chronos` | already FTSE | `A3.1 Fine-tuned LLM / foundation model` | explicit in the figure |
| `OneFitsAll` | Baselines | `A3.1` | "Zhou et al." in the figure |
| `StockTime`, `GPT4FTS`, `LLM-PS`, `CALF`, `Time-LLM` | FTSE | `A3.2 Edited / adapted LLM` | regrouped to the figure's leaf |
| `TimeHF`, `FinCast` | Baselines | `A3.3 Emerging foundation models` | explicit in the figure |
| `BiLSTM-ARIMA`, `CEEMDAN-Informer-LSTM` | Baselines | `A1.1 Hybrid LSTM` | the figure names them under FTSE → Deep Learning |
| `N-BEATS-GAN` | FTSS | `A2.1 GAN-based models` | it is a **forecasting** method; the figure places it under FTSE, not FTSS |

`Time-LLM` also moved from Baselines into FTSE, and `LSSDM` was already in FTSI but is not named in
the figure — it joins the score-based diffusion leaf without displacing anything.

### Leaves that are this repository's own

Marked *(beyond Figure 1)* in the README:

| Leaf | Contents | Reason |
|---|---|---|
| `A1.2 Ensemble, multi-source and feature-engineered forecasters` | 4 | the figure's Hybrid LSTM leaf does not cover ensembles or heterogeneous-data fusion |
| `A1.3 Other deep-learning forecasters` | 1 | `ORGAN-FVT` is a GAN-trained volatility *classifier*, not a generative forecaster |
| `C1.3 Combined / VAE-based models` | 4 | the figure's micro-level GAN and diffusion leaves have no VAE slot — but these 4 appear in the figure only as a **commented-out** "Combined Models" row (`ijcai26.tex:478`), so the leaf exists in the manuscript and was simply not rendered |
| `C1.4 Stylized-fact evaluation of synthetic series` | 2 | evaluation studies of synthesized series; no home in the figure |

The commented-out row is worth flagging to the authors: `Dogariu-Synth`, `VAE-GRU-MCMC`,
`FED2Port`, `WGAN-BiLSTM`, `TimeGAN-3D-CNN`, `FED`, `CFTNet`, `VAE-INN` and `Sideras` are all in
`ijcai26.bib` but not rendered in the figure. If that row is un-commented, the manuscript and this
README agree exactly on `C1.3`.

### One leaf left deliberately empty

`B1.2 Polynomial interpolation` is rendered as **empty**. The figure places `NDDP` there; no
verified record for NDDP could be resolved (the only secondary mention found describes it as
"an improved Newton interpolation polynomial" used inside `ALN`, without a citable standalone
record). Rather than fabricate an identifier, the leaf is shown empty and names what is missing.
This is the only leaf in the taxonomy with no entry.

### Resulting counts

| Section | Before | After |
|---|---|---|
| FTSE | 23 | **33** |
| FTSI | 13 | **13** |
| FTSS | 27 | **26** |
| **Literature Review** | 63 | **72** |
| General-purpose comparators | 10 | 9 |
| Related surveys | 1 | **26** |
| Adjacent | 7 | 8 |
| **Total** | 81 | **115** |

FTSE grows because the figure's foundation-model and diffusion branches are FTSE methods that the
old flat partition had parked under "Baselines". FTSS drops by one because `N-BEATS-GAN` moved to
FTSE. The Literature Review total rises by 9 — the ten moves, less `N-BEATS-GAN` leaving FTSS.

### Comparators section

The old "Forecasting Baselines & Foundation Models" section lost 9 of its 10 entries to the
taxonomy. It was rebuilt as **General-purpose Comparators** and repopulated with the standard
non-financial forecasters that FTSG papers report against: `N-BEATS`, `Informer`, `Autoformer`,
`PatchTST`, `DLinear`, `TimesNet`, `TimesFM`, `TimeMixer`, `MOMENT`. All nine are cited as arXiv
preprints because OpenAlex and Crossref hold no separate published record for them — the arXiv
comment states acceptance for PatchTST (ICLR 2023), Informer (AAAI 2021) and MOMENT (ICML 2024),
and that is recorded in each entry's Task line rather than asserted as a venue.

A **second** identifier error was caught here: `2302.10687` was written from memory as TimesFM and
is in fact *Boosting the Power of Kernel Two-Sample Tests* (Biometrika). The correct record is
**arXiv:2310.10688**, *A decoder-only foundation model for time-series forecasting*, Das, Kong, Sen
and Zhou. Same failure mode as the three arXiv IDs recorded in §4.

---

## 9. Post-restructure validation

Run against the generated README (115 entries):

| Check | Result |
|---|---|
| Entry headings | 115 |
| Duplicate nicknames | 0 |
| Entries with a resolvable link | 115 / 115 |
| DOIs registered in Crossref | 87 / 87 |
| arXiv identifiers whose abs-page title matches the README title | 27 / 27 |
| Non-DOI links | 28 (27 arXiv abs pages + 1 IJCAI proceedings page) |
| Entries carrying a publisher abstract | 83 / 115 |
| CJK residue | none |

One title was corrected during this pass: `Chronos-Arbitrage` was listed as *Large Language Models
for Time Series…* but the registered arXiv title is ***LLMs*** for Time Series: an Application for
Single Stocks and Statistical Arbitrage.

The remaining 32 entries without abstracts are publishers that deposit no abstract with the
metadata record (ACM, IEEE, Elsevier, Springer, SSRN, INFORMS). No abstract was written from a
title; those entries carry their link instead.
