"""Point-by-point response to the second internal review -> submission/response_to_review_2.docx.

    python paper/docx/response2.py

Numbers are macros from paper/generated/*.tex, expanded with the manuscript builder, so the response always matches
the manuscript. Section, table and figure numbers refer to the second revision of the Word manuscript and supplement.
The first-round response (response.py -> response_to_review.docx) is kept as it was sent and is no longer rebuilt.
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docx import ROOT, load_macros  # noqa: E402
from response import plain as _plain  # noqa: E402


def plain(text: str, macros: dict) -> str:
    t = _plain(text, macros)
    return t.replace("~", " ").replace("10^{-4}", "10⁻⁴").replace("10^{-3}", "10⁻³")

OUT = ROOT / "submission" / "response_to_review_2.docx"

INTRO = (r"We thank the reviewer for the second reading. We ran 96 further training runs (one diverged step size included): homes and the IID window "
         r"control trained to a plateau at $K=20$, 50 and 100 (up to 500 rounds); client Adam with kept state, client "
         r"SGD and server-side Adam (FedAdam) at $K=20$ and 100 with step sizes chosen on validation; the window control "
         r"in both further cohorts; client-only training with the same local fine-tuning as FedAvg; and seeds 45 and 46 "
         r"for the arms of the main comparisons that had three. All significance tests are now run-level with one Holm correction. "
         r"The title now names the training budget. Numbering below refers to the second revision: the former Fig.~3 is "
         r"now Fig.~4 and the cohort table is Table~4. Every number is generated from the result files.")

# (comment, response, where). Comments are quoted in short form.
ROWS = [
    ("Major 1: the headline is at 30 rounds; run homes and windows to a plateau at K = 20, 50, 100; plot error "
     "against rounds and MB; if the IID gap closes the title should say budget.",
     r"Done (Fig.~3, Table~2, Supplementary Table S5). At the plateau the gap was \RvPlhomesKtwentyGap{}, "
     r"\RvPlhomesKfiftyGap{} and \RvPlhomesKhundredGap{}\,\% with homes and \RvPlwindowsKtwentyGap{}, "
     r"\RvPlwindowsKfiftyGap{} and \RvPlwindowsKhundredGap{}\,\% with windows: more rounds removed "
     r"\RvPlRemovedLo{}--\RvPlRemovedHi{}\,\% of the 30-round gap, but the IID gap did not close. Because the budget "
     r"is a large part of the cost, the title is now ``Training budget and fragmentation, more than heterogeneity, "
     r"drive the accuracy cost of federated household load forecasting''. Fig.~2 now shows the plateau points and the "
     r"``upper bounds'' sentence is replaced by per-seed best and stop rounds; one $K=100$ run with homes reached the "
     r"500-round cap and is flagged as an upper bound.", "Title; Abstract; Section 4.2; Figs. 2-3; Table 2; Table S5"),
    ("Major 2: optimiser set-up (Adam restarts each round); repeat K = 20 and 100 with SGD, kept state or a server "
     "optimiser; explain the rise in rounds 4-5 of Fig. 3(a).",
     r"%OPTIMISER_RESPONSE% The early rise is explained in Section 4.4: its size was \RvRisewindow{}\,\% for the "
     r"window control and \RvRiseHomeLo{}--\RvRiseHomeHi{}\,\% for groupings of homes, and it coincides with "
     r"disagreement between client updates in rounds 2--5 (\RvRhoEarlyHomeLo{}--\RvRhoEarlyHomeHi{} against "
     r"\RvRhoEarlywindow{}). With Adam state kept the rise was \RvOptAdamKeepKtwozeroRise{}\,\% against "
     r"\RvOptAdamKtwozeroRise{}\,\% with restarts, so the restarts do not cause it.",
     "Sections 3.1, 4.3, 4.4; Tables S6-S7"),
    ("Major 3: one cohort; run the control in every cohort, quote ranges in the abstract, move Table S5 into the paper.",
     r"Done. The window control ran at $K=5$, 20 and 100 in both further cohorts (two seeds each). With 20 or more "
     r"clients it kept \RvCoShareBigLo{}--\RvCoShareBigHi{}\,\% of the gap in every cohort; this range is in the "
     r"abstract. At $K=5$ the gaps are small and change sign between cohorts, which the text states. The cohort table is "
     r"now Table~4 in the main text.", "Abstract; Section 4.5; Table 4"),
    ("Major 4: household-level tests over-confident; compare at seed level or with a mixed model; add a "
     "homes-versus-windows term; one correction rule; Section 4.4 uses uncorrected p-values.",
     r"All tests are now run-level: the pooled test MAE of each run, paired by seed (or Welch where seeds differ), "
     r"which equals the arm test of a mixed model with crossed household and run effects. One Holm correction covers "
     r"all \RvTsFamily{} comparisons (Supplementary Table S1) and every $p$ in the text is that value; household "
     r"shares are descriptive. Seeds 45 and 46 were added to the 30-round arms of the comparisons (five seeds each). The regression has a "
     r"homes-versus-windows term ($\RvRegTwohomes{}\times10^{-3}$\,kWh, CI \RvRegTwohomeslo{} to "
     r"\RvRegTwohomeshi{}); $\ln K$ and $H_{\mathrm{JS}}$ are correlated among home groupings ($r=\RvRegCorrKH{}$), and "
     r"within $K=20$ the heterogeneity term is not significant. With few runs per arm several differences are no longer "
     r"significant; the text says so rather than relying on household-level $p$-values.",
     "Sections 3.4, 4.1-4.8; Table S1"),
    ("Major 5: forecasting task weak (one step; FedAvg at K = 100 worse than persistence in MASE). Add day-ahead or "
     "limit the claims.",
     r"Claims are limited to one-step forecasts in the abstract, Section 4.5, the limitations and the conclusion; "
     r"day-ahead forecasting with rolling evaluation is named as future work.", "Abstract; Sections 4.5, 5, 6"),
    ("Major 6: unequal fine-tuning comparison; add client-only + FT; client-only rows in Table 1; one-home result in "
     "the abstract.",
     r"Client-only training was rerun with the same one-epoch per-home fine-tuning (\RvCliFTKfiveFT{} and "
     r"\RvCliFTKtwentyFT{}\,kWh at $K=5$ and 20), and both client-only rows are in Table~1. FedAvg + FT was better "
     r"than client-only + FT by $\RvTsFedCliFTKfiveDiff{}$ and $\RvTsFedCliFTKtwentyDiff{}\times10^{-4}$\,kWh in "
     r"\RvTsFedCliFTKfiveSeeds{} and \RvTsFedCliFTKtwentySeeds{} paired seeds ($p_{\mathrm{Holm}}$\,\RvTsFedCliFTKfiveP{} "
     r"and \RvTsFedCliFTKtwentyP{}). The abstract states that FedAvg + FT with one home per client "
     r"(\EonethreefedavghouseholdFT{}\,kWh) still loses to local models (\EonelocalMAE{}\,kWh).",
     "Abstract; Section 4.5; Table 1"),
    ("78-83 % qualified with K = 5: the paper says 58 % but Table 2 gives 55-56 %.",
     r"The share in the text was computed from unrounded gaps, the table gave rounded gaps, hence the mismatch. Table~2 "
     r"now has a column with the share computed from unrounded gaps, so text and table agree; with seeds 45 and 46 it is "
     r"\RvWinKfiveShare{}\,\% at $K=5$ and \RvWinShareLo{}--\RvWinShareHi{}\,\% at $K\ge20$. The cohort table had used "
     r"only seeds 42 and 43 of the main cohort; it now uses all seeds, as Table~2.",
     "Section 4.2; Tables 2 and 4"),
    ("“within 0.0008”, “rose steadily”, K = 50 best round 56 / stopped 76.",
     r"The range is given as \RvHetMaeLo{}--\RvHetMaeHi{}\,kWh. ``Rose steadily'' is replaced: the excess error was "
     r"\RvHetwindowExcess{}\,\% for the control and \RvHetExcessHomeLo{}--\RvHetExcessHomeHi{}\,\% for groupings of "
     r"homes, not ordered by $H_{\mathrm{JS}}$. Per-seed best and stop rounds are given for the 100-round runs "
     r"(Table S2: $K=50$ best \RvLongKfiftyBestSeeds{}, stopped \RvLongKfiftyRunSeeds{}) and for the plateau runs.",
     "Sections 4.2, 4.4; Tables S2, S5"),
    ("Traffic of the 100-round runs; S7 and Fig. S1 never cited.",
     r"The 100-round runs moved \RvLongKtwentyMB{}, \RvLongKfiftyMB{} and \RvLongKhundredMB{}\,MB at $K=20$, 50 and "
     r"100, and the plateau runs up to \RvPlhomesKhundredMBtotal{}\,MB; both are in Section 4.2. Fig.~S1 and every "
     r"supplementary table or its section are cited from the main text (the former Table S7 is now Table S9).", "Section 4.2"),
    ("§4.3: “every run qualified after the first round (14.3 MB)” contradicts Table 3.",
     r"Removed. Within 20\,\% the control settled after the first round in every seed (\RvHetwindowBtwozero{}\,MB) "
     r"while seeds of the groupings of homes needed \RvHetrandomBtwozeroLo{}--\RvHetclusterBtwozeroHi{}\,MB.",
     "Section 4.4"),
    ("§4.3: “the IID control needed more than random grouping” is now reversed; re-argue from seed ranges or drop.",
     r"Re-argued from seed ranges. Within 10\,\% the control needed the least traffic on average "
     r"(\RvHetwindowBonezero{}\,MB; seeds \RvHetwindowBonezeroLo{}--\RvHetwindowBonezeroHi{}\,MB). Among groupings of homes the means ran from "
     r"\RvHetshapesortedBonezero{}\,MB (shape-sorted, the most divergent) to \RvHetclusterBonezero{}\,MB (cluster), with "
     r"widely overlapping seed ranges (random \RvHetrandomBonezeroLo{}--\RvHetrandomBonezeroHi{}, cluster "
     r"\RvHetclusterBonezeroLo{}--\RvHetclusterBonezeroHi{}\,MB), so traffic did not follow $H_{\mathrm{JS}}$.",
     "Section 4.4; Table 3; Fig. 4"),
    ("Table 3 and Fig. 3 captions still describe the old definition.",
     "Both captions now describe the round after which validation MAE stays within ε for the rest of the run.",
     "Table 3; Fig. 4"),
    ("Table 3, B@5 %: a crossing only in the final round counts as reached.",
     r"A run that qualifies only in its last round now counts as not settled, in the code and in both captions. With "
     r"this rule the shape-sorted and $\alpha=1.0$ entries are no longer counted as reached; within 5\,\% "
     r"\RvHetwindowBfiveN{} of \RvHetwindowN{} control runs and \RvHetazerothreeBfiveN{} run each at $\alpha=0.3$ and "
     r"$\alpha=0.1$ settle, all near the end of the 30 rounds.", "Table 3; Fig. 4"),
    ("Title page: Dr Kalaavathi's ORCID dropped; corresponding author is a student address.",
     "All three ORCIDs are on the title page (0009-0004-7032-6847, 0009-0003-0735-264X, 0000-0003-4027-3137), and "
     "Dr Kalaavathi B (kalaavathi.b@vit.ac.in) is now the corresponding author.", "Title page; cover letter"),
    ("§5: restore the Briggs et al. detail “up to 500 epochs”.",
     "Restored, with the model size, and used as support for the budget explanation.", "Section 5"),
    ("Optional: stronger heterogeneity test (time-of-use homes or unequal client sizes).",
     "Not run. Time-of-use homes and unequal clients are named in the limitations and in future work.",
     "Sections 5 and 6"),
]


def main() -> Path:
    macros = load_macros()
    opt = OPTIMISER_RESPONSE
    rows = [(c, r.replace("%OPTIMISER_RESPONSE%", opt), w) for c, r, w in ROWS]
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(10.5)
    h = doc.add_heading("Response to the second internal review", level=1)
    h.runs[0].font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    doc.add_paragraph(plain(INTRO, macros))
    t = doc.add_table(rows=1, cols=4)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, head in zip(t.rows[0].cells, ["No.", "Comment", "Response", "Where"]):
        cell.text = ""
        cell.paragraphs[0].add_run(head).bold = True
    for i, (c, r, w) in enumerate(rows, 1):
        cells = t.add_row().cells
        for cell, val in zip(cells, [str(i), c, plain(r, macros), w]):
            cell.text = val
    for row in t.rows:
        for cell, wd in zip(row.cells, [0.4, 2.0, 3.4, 1.1]):
            cell.width = Pt(wd * 72)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    left = [i for i, (_, r, _) in enumerate(rows, 1) if "n/a" in plain(r, macros) or "\\" in plain(r, macros)]
    print(f"wrote {OUT} ({len(rows)} comments){'; unresolved in ' + str(left) if left else ''}")
    return OUT


OPTIMISER_RESPONSE = (
    r"Done at $K=20$ and 100 over 100 rounds (Section 4.3, Table S6). Step sizes were chosen on validation (Table S7): "
    r"client SGD \RvGridsgdBest{} (1.0 diverged) and FedAdam \RvGridfedadamBest{}; because 0.01 lay at the edge of the "
    r"grid, 0.03 was also run and is reported as not selected. Keeping Adam state made no difference "
    r"(\RvOptAdamKeepKtwozeroGap{} against \RvOptAdamKtwozeroGap{}\,\% at $K=20$, \RvOptAdamKeepKonezerozeroGap{} against "
    r"\RvOptAdamKonezerozeroGap{}\,\% at $K=100$). Client SGD was worse (\RvOptSGDKtwozeroGap{} and "
    r"\RvOptSGDKonezerozeroGap{}\,\%). FedAdam did not help at $K=20$ (\RvOptFedAdamKtwozeroGap{}\,\%) but narrowed the "
    r"one-home-per-client gap to \RvOptFedAdamKonezerozeroGap{}\,\% (\RvOptFedAdamOtherKonezerozeroGap{}\,\% at step 0.03), "
    r"in every paired seed though not significantly ($p_{\mathrm{Holm}}$\,\RvTsFedAdamKhundredP{}). No optimiser closed "
    r"the gap; the text now says that a server-side optimiser recovers part of the budget cost.")

if __name__ == "__main__":
    main()
