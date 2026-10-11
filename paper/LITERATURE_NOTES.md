# Literature notes: bibliography audit and positioning

Compiled 2026-10-09 for the manuscript "Measuring what heterogeneity costs: a controlled, communication-priced study of federated LSTM load forecasting on London smart meters".

Output bibliography: `paper/refs_verified.bib`. It has 68 entries: the 39 original entries with their keys unchanged, plus 29 new ones. Every `\cite` key in `manuscript.tex` resolves against it.

Sources used for checking:
- Crossref API, for DOIs, titles, authors, volumes and pages.
- PMLR, NeurIPS and MLSys proceedings pages, for conference volumes and pages.
- The arXiv API.
- DataCite.
- The London Datastore page.
- Full-text PDFs for Briggs 2022, Fekri 2022, Taïk 2020, Gholizadeh 2022, FedProx and Briggs 2020.

ML conference papers (ICLR, ICML, NeurIPS, MLSys, AISTATS) do not have DOIs, so their entries carry a proceedings or arXiv URL instead.

---

## 1. Verification status of the original `refs.bib` (39 entries)

**Result: all 39 verified. None is marked UNVERIFIED. 37 now carry a DOI.** The two without a DOI are Holm 1979 (JSTOR only) and the London Datastore dataset. Seven of the ML papers carry a URL instead of a DOI.

| Key | Status | DOI / identifier | Corrections made |
|---|---|---|---|
| fernndez2022 | Verified | 10.1016/j.apenergy.2022.119915 | Full author names |
| fekri2022 | Verified | 10.1016/j.ijepes.2021.107669 | Full names; full journal name |
| gholizadeh2022 | Verified | 10.1016/j.iot.2021.100470 | Full names |
| yang2023 | Verified | 10.1016/j.epsr.2022.108830 | Full names |
| liu2023 | Verified | 10.1016/j.ijepes.2023.109172 | Journal abbreviation "IJEPES" expanded |
| fekri2023 | Verified | 10.1016/j.ijepes.2023.109285 | Journal expanded |
| qu2023 | Verified | 10.26599/BDMA.2022.9020043 | `volume={6(4)}` split into volume 6, number 4; en-dash fixed to `--` |
| wang2023 | Verified | 10.1016/j.enbuild.2023.113215 | Full names |
| khalil2023 | Verified | 10.1016/j.cose.2023.103211 | Full names |
| widmer2023 | Verified | 10.1016/j.egyai.2023.100253 | Full names |
| dogra2023 | Verified | 10.1016/j.scs.2023.104815 | **Author field was broken** ("Dogra and S. and others"). Now Dogra, Atharvan; Anand, Ashima; Bedi, Jatin |
| xiao2023 | Verified | 10.1016/j.enbuild.2023.113178 | **Author field was broken**. Now Xiao, Cao, Fang, Wang, Wang. Title is "...with selected-shared-private mechanism", without the "a" |
| dab2023 | Verified | 10.1016/j.enbuild.2023.113550 | **Author field was broken**. Six authors restored |
| sarmento2024 | Verified | 10.1016/j.iot.2023.101050 | **Author field was broken**. Seven authors restored |
| abdulla2024 | Verified | 10.1016/j.segan.2024.101342 | **Journal and volume were garbled** (journal "Sustainable Energy", volume "Grids and Networks", pages 38). Now *Sustainable Energy, Grids and Networks*, vol. 38, art. 101342 |
| singh2024 | Verified | 10.1016/j.knosys.2024.111967 | Full names |
| manzoor2024 | Verified | 10.1016/j.iot.2024.101376 | Full names |
| mao2024 | Verified | 10.1016/j.knosys.2023.111338 | Full names |
| aurangzeb2024 | Verified | 10.1016/j.egyr.2024.03.028 | Title uses "bi-directional"; en-dash fixed |
| huang2024 | Verified | 10.1016/j.asej.2024.102724 | Title fixed to "...with **considering** electricity data privacy..."; issue 6 added |
| mostafanezhad2025 | Verified | 10.1016/j.segan.2025.101794 | **Journal and volume garbled**. Now SEGAN vol. 43, art. 101794. Title has no hyphen: "poison attack tolerant" |
| kaur2025 | Verified | 10.1016/j.segan.2025.102021 | **Journal and volume garbled**. Now SEGAN vol. 44, art. 102021 |
| shao2026 | Verified | 10.1109/TSG.2025.3628876 | Volume and number split; en-dash fixed |
| das2026 | Verified | 10.1038/s41598-026-53020-6 | None |
| wang2026 | Verified | 10.1016/j.eswa.2025.130417 | Full names |
| tibermacine2026 | Verified | 10.3389/fenrg.2026.1854654 | **Author field was broken**. Seven authors restored |
| he2026 | Verified | 10.1016/j.jiixd.2026.04.009 | **Author field was wrong**: it has only 3 authors (He, Zeyu; Zhang, Haonan; Liu, Yining), not "and others". Volume and number split |
| mcmahan2017fedavg | Verified (PMLR 54:1273–1282) | URL | URL added |
| li2020fedprox | Verified (MLSys 2:429–450) | URL | URL added |
| karimireddy2020scaffold | Verified (PMLR 119:5132–5143) | URL | URL added |
| hsu2019dirichlet | Verified (arXiv:1909.06335) | 10.48550/arXiv.1909.06335 | arXiv DOI added |
| kairouz2021advances | Verified | 10.1561/2200000083 | DOI added |
| hochreiter1997lstm | Verified | 10.1162/neco.1997.9.8.1735 | DOI added |
| lin1991divergence | Verified | 10.1109/18.61115 | DOI added |
| wilcoxon1945 | Verified | 10.2307/3001968 | DOI added |
| holm1979 | Verified (Scand. J. Stat. 6(2):65–70) | No DOI; JSTOR 4615733 | URL and note added |
| zhu2019leakage | Verified (NeurIPS 32) | URL | Proceedings URL added. NeurIPS metadata gives no page numbers |
| ukpn2015lcl | Verified (dataset page exists; title, author, licence and content match) | URL | Note added. **Caveat:** the Datastore page shows no publication date, so the year 2015 could not be confirmed |
| kong2019stlf | Verified | 10.1109/TSG.2017.2753802 | DOI added |

All 27 references of the original bibliography are confirmed to exist with matching titles and venues (via Crossref). Indexing status (Scopus/WoS) was not checked.

---

## 2. New entries (Task B), with summaries

Items 2 and 5 were already in the bibliography as `fekri2022` and `gholizadeh2022`, so they were not duplicated. If you prefer the suggested key `fekri2022rnn`, cite `fekri2022`; it is the same paper.

### Federated household load forecasting

- **briggs2022**: Briggs, Fan, Andras, *IEEE OAJPE* 9:573–583 (2022), doi 10.1109/OAJPE.2022.3206220.
  - **Dataset (verified from the full text):** Low Carbon London, a sample of 100 households at 30-minute resolution, with Heathrow weather.
  - **Method:** a 2-layer LSTM trained with FedAvg, plus FL+HC (hierarchical clustering of clients) and local fine-tuning (LFT) personalisation. Compared against centralised and fully local training.
  - **Key finding:** plain FL was worse than localised learning (mean RMSE 7.6% worse; FL+HC 4.7% worse). FL or FL+HC followed by local fine-tuning was about 4.5% better than local, at about 10x less computation.
  - **Relevance:** this is the **closest prior work** (same dataset, same model family). It already reports "FL worse than local". It does not measure bytes communicated, and it does not control heterogeneity.
- **fekri2022** (item 2): *IJEPES* 137:107669.
  - **Data:** London Hydro (Ontario, Canada), 19 residential consumers, hourly.
  - **Method:** LSTM trained with FedSGD vs FedAvg, with 6 clients per round. Also studies clients joining dynamically.
  - **Finding:** FedAvg beats FedSGD, needs fewer rounds, and is comparable to or better than per-meter and central LSTMs.
  - **Communication:** counted in rounds only, not bytes.
- **taik2020** (item 3): Taïk and Cherkaoui, IEEE ICC 2020, doi 10.1109/ICC40277.2020.9148937.
  - **Data:** Pecan Street Dataport, 200 Texas houses (180 train, 20 held out), hourly, Jan–Mar 2019.
  - **Method:** LSTM with FedAvg in TensorFlow Federated, followed by local personalisation.
  - **Communication:** they define a "network load gain" G = 1 − L_FL/L_central. With a 1.9 Kb model vs 16 Mb of data they report 90–97% savings. **This is an analytic payload comparison of FL vs raw-data upload. It is prior art for "pricing communication".**
- **savi2021** (item 4): Savi and Olivadese, *IEEE Access* 9:95949–95969, doi 10.1109/ACCESS.2021.3094089.
  - **Method:** edge-based FL of LSTMs with calendar and weather features. Users are grouped by K-means on consumption or by socio-economic affinity.
  - **Finding (from the abstract):** RMSE similar to a centralised baseline, up to 10x less training time, and **"up to 50 times less exchanged data"** when sampling is finer than hourly. **The dataset could not be verified** because the full text was not retrievable. Do not state which dataset it used without checking the PDF.
- **gholizadeh2022** (item 5):
  - **Data:** EPCOR (Edmonton), residential buildings at 75 locations. The data is not public.
  - **Method:** LSTM + FC layers with FedAvg. Clients are clustered by their locally tuned hyperparameters. Covers individual and aggregate load, compared with centralised and local learning.
  - **Finding:** best individual RMSE was 0.117 kWh. However, **"on average, local learning performs best"** for individual load prediction. FL was comparable to local and better than centralised. A second prior instance of "FL ≤ local".
- **briggs2020ccai**: Briggs et al., NeurIPS 2020 CCAI workshop (arXiv 2012.07449). A position paper. It states *qualitatively* that sending a large weight vector instead of half-hourly readings "represents a significant increase in bandwidth" on 2G smart-meter links. **It makes no measurement.** This is a useful hook: your manuscript measures what Briggs et al. only asserted.
- **lu2024fds**: Lu et al., *IEEE TSG* 15(4):4271–4283 (2024). Federated domain separation for non-IID household loads. Splits the knowledge in client updates into a shared part and a "contaminating" part that is kept local. Personalised FL.
- **husnoo2024**: *IEEE TSG* 15(2):2044–2055. A secure FL framework for residential STLF.
- **he2023hfl**: *IEEE TSG* 14(6):4409–4423. A hierarchical, privacy-preserving FL framework for residential STLF.
- **duttagupta2023**: SmartGridComm 2023. A lightweight fully connected network trained with FL, using the London households data with K-means (18 clusters per the arXiv version). RMSE 0.17; 50 mWh training/inference energy on an Arduino.
- **rahman2025multihop**: *IEEE IoT J.* 12(18) (2025). Meta-learning personalised FL over multihop smart-meter networks, with latency optimisation.
- **bose2023pl**: arXiv 2309.13194. FL with personalisation layers on NREL ComStock (commercial buildings). Reports per-round payload size in MB (about 2.36 MB down to 150 KB) but **does not compare against raw-data upload**.
- **chen2026init**: arXiv 2608.27791 (Aug 2026, preprint). Documents "structured heterogeneity" in clients' load data. Proposes pretrained global initialisation and SLIAvg sequential local initialisation to reduce client drift.

### Household STLF baselines (item 10)

- **kong2019stlf**: already present and verified.
- **shi2018pdrnn**: Shi, Xu, Li, *IEEE TSG* 9(5):5271–5280 (2018), doi 10.1109/TSG.2017.2686012. A pooling deep RNN that pools neighbouring households' data to reduce overfitting in household-level forecasting.

### FL optimisation, personalisation, fairness and non-IID (item 6)

- **acar2021feddyn**: FedDyn, ICLR 2021. A per-client dynamic linear-plus-quadratic regulariser that aligns stationary points of the local and global objectives.
- **wang2020fednova**: FedNova, NeurIPS 33:7611–7623. Normalised averaging that removes objective inconsistency caused by heterogeneous numbers of local steps.
- **reddi2021fedopt**: FedOpt (FedAdam, FedAdagrad, FedYogi), ICLR 2021. Uses adaptive server optimisers.
- **fallah2020perfedavg**: Per-FedAvg, NeurIPS 33:3557–3568. MAML-style personalised FL.
- **li2021ditto**: Ditto, ICML 2021, PMLR 139:6357–6368. Personalised models regularised toward the global model by λ‖v_k − w‖²; targets fairness and robustness.
- **li2020qffl**: q-FFL / q-FedAvg, ICLR 2020. Reweights clients so that high-loss clients get more weight, giving fairer accuracy across clients.
- **zhao2018noniid**: arXiv 1806.00582. Accuracy loss under non-IID data relates to the earth mover's distance between class distributions. Proposes sharing a small global subset of data.
- **hsieh2020quagmire**: ICML 2020, PMLR 119:4387–4398. Skewed label partitions break decentralised training, and batch normalisation is especially fragile. Proposes SkewScout.
- Already present: li2020fedprox, karimireddy2020scaffold, hsu2019dirichlet.

### Communication, privacy and framework (items 7, 8, 11)

- **konecny2016comm**: arXiv 1610.05492. Structured and sketched updates to cut uplink cost.
- **zhu2019leakage**: already present. Deep Leakage from Gradients.
- **geiping2020inverting**: NeurIPS 33:16937–16947. Inverts gradients via cosine-similarity matching; shows that even multi-step FedAvg updates leak inputs.
- **beutel2020flower**: arXiv 2007.14390 (11 authors, verified).

### LCL dataset documentation (item 9)

- **ukpn2015lcl** (existing): London Datastore, "SmartMeter Energy Consumption Data in London Households", UK Power Networks, CC BY 4.0.
  - About 5,567 households, half-hourly kWh, Nov 2011 – Feb 2014.
  - About 1,100 households were on the dynamic ToU tariff in 2013.
- **ukds2016lcl**: UK Data Service SN 7857, doi 10.5255/UKDA-SN-7857-1, "Low Carbon London Project: Data from the Dynamic Time-of-Use Electricity Pricing Trial, 2013" (Schofield et al.). This is the citable, DOI-bearing archival release.
- **schofield2014lclA3**: LCL Learning Lab Report A3, "Residential consumer responsiveness to time-varying pricing" (Imperial College / UKPN, 2014). This is the canonical trial-design report.
- Other supporting sources, not added to the bib:
  - Schofield's 2015 PhD thesis (Imperial). It reports 5,533 participants, of whom 1,119 were on dToU.
  - CIRED 2015 paper "Experimental validation of residential consumer responsiveness to dynamic time-of-use pricing". **Its DOI could not be verified**, so it was not added.
  - Household counts differ across sources (5,567 / 5,533 / 4,756 after cleaning). State whichever count your preprocessing yields.

---

## 3. Novelty positioning

### 3a. Adaptive or heterogeneity-aware FedProx μ (for "HA-FedProx")

The proposed rule is D = Σ p_k‖Δ_k − Δ̄‖² / Σ p_k‖Δ_k‖².

Closest prior work found:

1. **FedProx itself (li2020fedprox).**
   - **Adaptive-μ heuristic (verified in arXiv v5, Sec. 5.3.3 / App. C.3.3, Figs. 3 and 11):** "increase μ by 0.1 whenever the loss increases and decrease it by 0.1 whenever the loss decreases for 5 consecutive rounds". This is a *loss-driven* adaptive μ, not a dispersion-driven one.
   - **B-local dissimilarity (Definition 3):** B² = E_k‖∇F_k‖² / ‖∇f‖². **Important algebraic link:** with Δ̄ = Σ p_k Δ_k,

     D = 1 − ‖Δ̄‖² / Σ p_k‖Δ_k‖² = 1 − 1/B̂²,

     where B̂ is FedProx's B-dissimilarity evaluated on the observed model updates instead of on gradients. Reviewers may notice this. Present D as a *bounded [0,1], update-space, per-round empirical estimate of FedProx's own B-dissimilarity*. Do not present it as a new heterogeneity measure.
2. **Gradient diversity (yin2018diversity, AISTATS 2018).** Σ‖g_i‖² / ‖Σ g_i‖² is the same ratio family, used to explain mini-batch speed-up limits. It is not used to set μ.
3. **HAPI-FedProx (mhou2025hapi, SACAIR / CCIS 2025, doi 10.1007/978-3-032-11733-5_17).** **The most direct prior art.** It adapts FedProx's proximal term from a "Heterogeneity-Aware Performance Index" that measures local-vs-global model differences. Evaluated on CIFAR-10 only. Abstract-level numbers: slightly worse than FedAvg under IID data, and +2 pp under strong non-IID data. **The exact HAPI formula could not be verified** (paywalled), so do not claim your rule differs from it in a specific way without reading the chapter.
4. **Yu et al. (yu2025pflminimax, ICML 2025).** Shows theoretically how to choose the personalisation and regularisation degree in a FedProx-type objective as a function of statistical heterogeneity. This is a theoretical, not online, choice.
5. **FedAdp (wu2021fedadp).** Uses angles between local and global gradients to set *aggregation weights* rather than μ. Related "update-geometry-aware" prior art.
6. **Search result:** no paper was found that sets μ from the *normalised dispersion of client updates* exactly as written above.

**Recommended claim:** "a simple, parameter-free per-round instantiation of FedProx's B-dissimilarity on model updates, used to schedule μ". Cite FedProx's loss-based heuristic, HAPI-FedProx and gradient diversity as the closest prior work. Avoid claiming to be "the first adaptive-μ FedProx", because that claim is false given FedProx Sec. 5.3.3 and HAPI-FedProx.

### 3b. Communication cost vs raw-data upload, and FL losing to local models (item 12)

**Prior work that compares communication against raw-data upload:**
- **taik2020** gives an *analytic* network-load gain: 90–97% in favour of FL for 1.9 Kb models vs 16 Mb of hourly data, counting hops.
- **savi2021** reports up to 50x less exchanged data than centralised training at sub-hourly resolution (abstract-level).
- **briggs2020ccai** argues the *opposite*, qualitatively: FL increases bandwidth relative to half-hourly readings.
- **bose2023pl** reports per-round MB but has no raw-data comparison.
- **fekri2022** and FedTrees-style papers count rounds, not bytes.
- **No 2024–2026 paper was found** that *measures* end-to-end bytes (uplink plus downlink over all rounds) against the raw-data upload for household load forecasting on LCL, or that reports a break-even point.

So "communication-priced" is defensible as novel *in measurement and controlled design*, but not as a first-ever comparison. Cite Taïk (analytic gain) and Savi (50x) as prior estimates. Note that the two reach opposite conclusions from Briggs 2020's claim, and that the sign depends on model size × rounds vs data resolution × horizon. This is exactly the quantity your study prices.

**Prior evidence that FL is worse than local models for households:**
- **briggs2022 (LCL, LSTM):** FedAvg mean RMSE 7.6% worse than local; FL+HC 4.7% worse. Personalisation recovers about +4.5%.
- **gholizadeh2022:** local learning best on average for individual households.
- **widmer2023:** model personalisation was best in 59.1% of cases (from a search-result summary; the full text was not read).
- **das2026:** FedAvg was the weakest federated method in their 22-client benchmark.
- Counter-evidence: **fekri2022** found FedAvg comparable to or better than local on 19 Ontario homes.

The "FL can be worse than local" finding is therefore **already published** (notably Briggs 2022 on the same dataset). Frame your contribution as **quantifying when and by how much**, under controlled heterogeneity, with paired statistics (Wilcoxon/Holm) and a communication price. Do not frame it as discovering the effect.

---

## 4. Unverified or partially verified items

- **Savi and Olivadese 2021: the dataset used.** Not verified. The bib entry itself is verified.
- **HAPI-FedProx: the exact definition of the HAPI index and the μ update.** Not verified (paywalled). The CCIS volume number is also not verified; it is noted in the entry.
- **ukpn2015lcl: the year 2015.** Not stated on the Datastore page.
- **widmer2023: the "59.1%" result.** Taken from a search-result summary, not the full text.
- **ICLR papers (acar2021feddyn, reddi2021fedopt, li2020qffl).** The venue is confirmed via the arXiv comment fields and the iclr.cc / mlanthology listings. The OpenReview pages could not be fetched (bot check), so the entries use arXiv URLs.
- **CIRED 2015 LCL paper.** Its DOI could not be found, so it was not added.
- **Indexing (Scopus/WoS) of the original 27 references.** Not checked.

## Revision checks (2026-10-11)

- **briggs2022, Table II (verified from arXiv 2105.13325v2):** two-layer LSTM with 20 hidden units per layer, hourly sequences of 6/12/24 steps, with and without weather; up to 500 epochs with early stopping. Mean RMSE: centralised 0.0205, localised 0.0196, so centralised was 4.8 % *worse* than localised. FL was 7.6 % worse, FL+HC 4.7 % worse, and FL→LFT 4.3 % better. RMSE is averaged over clients for the distributed arms and over all sequences for the centralised arm.
- **taik2020 (verified from arXiv 2201.11248):** two LSTM hidden layers of 200 neurons each. The text says "the size of the model is 1,9Kb and the size of the used data is 16Mb". Two such layers hold about 4.8×10^5 weights, which is about 1.9 MB in 32-bit floats, so the reported unit is probably a slip. The quotation is correct, though.
- **Local SGD:** stich2019localsgd (ICLR 2019), woodworth2020localsgd (ICML 2020, PMLR 119:10334–10343) and woodworth2020minibatch (NeurIPS 2020; the proceedings page lists no page range). All were verified on the proceedings or arXiv pages.
