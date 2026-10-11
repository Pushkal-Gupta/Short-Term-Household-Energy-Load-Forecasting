"""Point-by-point response to the internal review (35 comments) -> submission/response_to_review.docx.

    python paper/docx/response.py

Numbers are macros from paper/generated/*.tex, expanded with the manuscript builder, so the response always matches
the manuscript. Section, table and figure numbers refer to the revised Word manuscript (manuscript_template.docx) and
the supplementary material (supplementary_material.docx).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docx import ROOT, expand, load_macros  # noqa: E402

OUT = ROOT / "submission" / "response_to_review.docx"

INTRO = (r"We thank the reviewer for a careful reading. We ran 86 further training runs to answer the major points: an "
         r"IID control that deals training windows rather than homes to clients, runs of up to 100 rounds, a "
         r"load-shape grouping, a client-only baseline, calendar inputs, a proximal grid that includes $\mu=0$, two "
         r"more seeds for the heterogeneity, participation and precision settings, and two further cohorts. The "
         r"title is softened to ``Fragmentation outweighs heterogeneity in the accuracy cost of federated household "
         r"load forecasting''. The 85\,\% traffic claim is withdrawn and the adaptive proximal rule is moved to the "
         r"supplement. Every number below is generated from the result files, as in the manuscript.")

# (comment, response, where). Comments are quoted in short form.
ROWS = [
    ("Overall: headline claim not proven; fragmentation, heterogeneity and budget are entangled. Add the controls or "
     "soften the title.",
     r"Both. The IID control (windows dealt at random) kept \RvWinShareLo{}--\RvWinShareHi{}\,\% of the gap "
     r"at $K\ge20$ with clients that do not differ, so fragmentation is the larger cost; the regression now finds a "
     r"significant but smaller heterogeneity term, and longer training narrows the gap. The title was softened from "
     r"``not heterogeneity'' to ``outweighs heterogeneity'', and the abstract states all three effects.",
     "Title; Abstract; Sections 4.2-4.3"),
    ("Minor: school-level affiliation only.", "Done: School of Computer Science and Engineering, Vellore Institute of "
     "Technology.", "Title page"),
    ("Minor: remove the personal Gmail address.", "Removed; only institutional addresses remain.", "Title page"),
    ("Check: FedAvg with one client is centralised by construction; present as a sanity check.",
     r"Now described as a sanity check of the implementation, labelled so in Table 1, and removed from the "
     r"contributions.", "Section 4.1; Table 1"),
    ("Check: the traffic result holds for FedAvg only and rests on a first-crossing count.",
     r"The 85\,\% claim is withdrawn. With five seeds the increase at $\varepsilon=10\,\%$ is \RvHetBepsIncrease{}\,\%, "
     r"and the IID control needs more traffic than random grouping (\RvHetwindowBonezero{} against "
     r"\RvHetrandomBonezero{}\,MB), so the difference is within seed variation. The abstract no longer makes a traffic "
     r"claim for heterogeneity.", "Abstract; Section 4.3"),
    ("Check: state the configuration or range of the 66 % to 48 % claim.",
     r"The abstract now states $K=20$, $\alpha=0.3$ and uses the 1\,\% margin "
     r"(\RvHarmFedModerateMargin{}\,\% to \RvHarmFedModerateFTMargin{}\,\%); Section 4.7 gives the range over "
     r"groupings and $K=100$.", "Abstract; Section 4.7; Table 4"),
    ("Major 6 (1): centralised beats persistence by only 8.9 %; add calendar inputs or a stronger model; report MASE.",
     r"Calendar inputs were added to centralised, local and FedAvg ($K=20$, 100) arms. They did not help (centralised "
     r"\EtwocentralizedMAE{} to \RvCalCentralisedmaecal{}\,kWh; FedAvg $K=100$ worse). MASE is reported: centralised "
     r"\RvCalCentralisedmase{}, persistence \RvMASEPersistence{}. We state in the limitations that single-home "
     r"forecasts at 30 minutes are close to persistence, so absolute gains are small.", "Section 4.4; Table S8"),
    ("Minor: notation t and k used twice.", r"$t$ is now the half-hour, $r$ the round, $k$ the client and $G$ the "
     r"number of $k$-means clusters.", "Section 3"),
    ("Major 6 (2): one cohort; add cohorts.",
     r"Two further cohorts of 100 homes (selection seeds 43 and 44, sharing three or four homes with the main cohort). "
     r"The gap grew with $K$ in all three; at $K=100$ it ranged from \RvCohortGapKonezerozeroLo{} to "
     r"\RvCohortGapKonezerozeroHi{}\,\%. Local-only beat centralised in one cohort, which we report.",
     "Section 4.4; Table S5"),
    ("Minor: local-only budget.", "At most 30 epochs with early stopping on validation MAE, patience 4.", "Section 3.3"),
    ("Minor: how Dirichlet and cluster groupings map homes; equal sizes?",
     r"Each grouping is now described step by step; all clients hold exactly $100/K$ homes.", "Section 3.3"),
    ("Major 3: heterogeneity range narrow; composite H dominated by mean load; justify H, one index.",
     r"The composite is dropped (its level term made up \RvHcvShare{}\,\% of it). $H_{\mathrm{JS}}$ (base 2, so bounded "
     r"by 1, not ln 2) is used throughout, with $H_{\mathrm{CV}}$ reported separately. A shape-sorted grouping was "
     r"added ($H_{\mathrm{JS}}=\RvHetshapesortedHJS{}$, the widest split available, with unchanged error "
     r"\RvHetshapesortedMAE{}\,kWh); the reachable range remains small, and the discussion limits the conclusion "
     r"accordingly.", "Section 3.3; Table 3; Section 5"),
    ("Minor: state the raw-data encoding; show 2-4 bytes and CSV.",
     r"12 bytes per reading is stated, and the comparison is given at 2 bytes (\RvRawTwo{}\,MB), 4 bytes "
     r"(\RvRawFour{}\,MB), 12 bytes (\RvRawTwelve{}\,MB) and CSV (\RvRawCsv{}\,MB). Thirty rounds at $K=20$ still move "
     r"more (\EthreefedavgiidMB{}\,MB).", "Sections 3.3 and 4.6"),
    ("Major 1: gap may be under-training; plot per-round validation MAE per K, report chosen round, run K >= 20 to a "
     "plateau, state whether Adam is reset.",
     r"Adam is reset at every local update (stated). Steps per round and the chosen round are tabulated, and the "
     r"curves plotted (Table S2, Fig. S1). Runs of up to 100 rounds with patience 25 narrowed the gap to "
     r"\RvLongKtwentyGap{}, \RvLongKfiftyGap{} and \RvLongKhundredGap{}\,\% at $K=20$, 50, 100. $K=50$ plateaued; "
     r"at $K=20$ and 100 the best round was still near the end, which we state.",
     "Sections 3.1, 3.4, 4.2; Fig. 2; Table S2; Fig. S1"),
    ("Minor: what the seeds vary.", "Initialisation, grouping or Dirichlet draw, client sampling and stragglers, and "
     "mini-batch order; the cohort is fixed.", "Section 3.4"),
    ("Check: WAPE never reported; MAPE* unexplained.", "WAPE is reported for every arm in Table 1; MAPE is dropped "
     "from the paper and from Fig. 1, which now lists MAE, RMSE, WAPE and MASE.", "Table 1; Fig. 1"),
    ("Minor: seeds in the per-household tests; Holm family.",
     "Household MAE is averaged over seeds before testing; Holm is applied over all rows of Table S1.",
     "Section 3.4; Table S1"),
    ("Check: 'every LSTM beat both naive references' false for K = 100 MAE.",
     r"Corrected: FedAvg with one home per client is worse than persistence in MAE (\EonethreefedavghouseholdMAE{} "
     r"against \PersistenceMAE{}\,kWh).", "Section 4.1"),
    ("Minor: p-values lack '='; hyphens instead of minus signs.", "All p-values now read '= value' or '< 10^-4', and "
     "tables use the minus sign.", "Throughout"),
    ("Major 2: fragmentation and heterogeneity confounded; add IID control; H_JS for every K, log base, CIs.",
     r"The window control was added for $K=5$--100 (Table 2, Fig. 2), with $H_{\mathrm{JS}}$ for every $K$ and both "
     r"partitions. The regression now uses $n=\RvRegN{}$ runs and reports 95\,\% CIs: $\ln K$ "
     r"\RvRegLogK{}$\times10^{-3}$ (\RvRegLogKlo{} to \RvRegLogKhi{}), $H_{\mathrm{JS}}$ \RvRegHJS{}$\times10^{-3}$ "
     r"(\RvRegHJSlo{} to \RvRegHJShi{}). Both are significant; over the tested ranges the $K$ effect is about four "
     r"times the heterogeneity effect (\RvRegEffK{} against \RvRegEffH{}\,kWh). Base-2 logarithms are stated.",
     "Sections 3.3, 4.2; Table 2; Fig. 2"),
    ("Minor: remove titles inside Figs. 2-4.", "Removed; panels carry only (a), (b).", "Figures"),
    ("Major 4: B@eps looks like a threshold artefact; show curves, vary eps, add seeds and intervals; FedAvg only.",
     r"Curves are shown (Fig. 3a), $\varepsilon\in\{5,10,15,20\}\,\%$ with per-seed points (Fig. 3b), five seeds, 95\,\% "
     r"intervals, the number of seeds that reached each threshold, and a threshold-free excess-error measure. The "
     r"claim is withdrawn as above and limited to FedAvg.", "Sections 3.3, 4.3; Table 3; Fig. 3"),
    ("Major 7: boundary optimum for mu; widen the grid to 0 or move HA-FedProx out of the contributions.",
     r"Both. The grid now includes $\mu=0$ and $10^{-4}$; $\mu=0$ (FedAvg) is best (\RvTuneFedAvgVal{} against "
     r"\RvTunefedproxBestPosVal{}\,kWh for the best positive $\mu$). HA-FedProx is removed from the contributions and reported in Supplementary S3.",
     "Section 4.5; Tables S3-S4"),
    ("Minor: give n for the Spearman test.", r"$n=\RvHetSpearmanN{}$ runs is given (and $n$ for the $\rho$ test in "
     r"S3).", "Section 4.3"),
    ("Minor: why are 16-bit updates and partial participation better?",
     r"Rechecked with five seeds; both are slightly better (16-bit for \RvPrecfponesixBetter{}\,\% of households, "
     r"median \RvPrecfponesixMed{}\,kWh). We describe this as a plausible mild regularisation effect below 1\,\% of "
     r"MAE, not a gain to exploit.", "Section 4.6"),
    ("Check: treat non-significant results consistently.", r"All results with $p_{\mathrm{Holm}}\ge0.05$ are now "
     r"described as not significant, without implying equivalence (stated in Section 3.4).", "Sections 3.4, 4"),
    ("Check: add results by consumption quintile.", r"Added (Table S6); the text gives the top-quintile medians.",
     "Section 4.7; Table S6"),
    ("Minor: use a margin for 'harmed'.", r"A 1\,\% margin is used, with the share helped by more than 1\,\% "
     r"(Table 4).", "Section 4.7; Table 4"),
    ("Minor: zoom the x-axis of the benefit CDF.", "Zoomed to the 2nd-98th percentile range; the figure moved to the "
     "supplement to meet the page limit.", "Fig. S2"),
    ("Minor: cite the local-SGD literature and test the mechanism with the controls.",
     "Stich (2019), Woodworth et al. (2020, two papers) and Reddi et al. (2021) are cited; the IID control and the "
     "budget runs test the mechanism.", "Sections 2.3, 5"),
    ("Check: Briggs et al. Table II had centralised worse than localised; discuss.",
     r"Confirmed and discussed: their models (two 20-unit layers, hourly sequences of 6-24 steps, up to 500 epochs) "
     r"and per-sequence averaging differ. We also report that local-only beat centralised in one of our new cohorts.",
     "Sections 2.1, 5"),
    ("Check: RMSE claim not supported at K = 50 and 100.", "The RMSE statement was removed.", "Section 5"),
    ("Minor: Taïk's '1.9 kb' looks like a unit slip.", "Stated in the text with the weight count (about 4.8 x 10^5 "
     "weights, 1.9 MB); the template has no footnotes, so it is a sentence.", "Section 5"),
    ("Major 5: pooled clients weaken the privacy premise; add a client-only baseline; who is trusted.",
     r"%CLIENTONLY_RESPONSE%", "Sections 3.3, 4.4, 5"),
    ("Minor: insert the repository URL.", "Inserted (https://github.com/Pushkal-Gupta/Short-Term-Household-Energy-"
     "Load-Forecasting); the repository will be made public before submission.", "Data availability"),
]

CLIENTONLY = (r"Added for $K=5$ and $K=20$ (each client trains alone, with early stopping, on the homes it pools). "
              r"Plain FedAvg was not significantly better than client-only training (better for "
              r"\RvClientKfiveFedBetter{} and \RvClientKtwentyFedBetter{}\,\% of households, $p$ \RvClientKfiveFedP{} and "
              r"\RvClientKtwentyFedP{}); with local fine-tuning it was (\RvClientKfiveFTBetter{} and "
              r"\RvClientKtwentyFTBetter{}\,\%, $p$ \RvClientKfiveFTP{} and \RvClientKtwentyFTP{}). This is now in the "
              r"abstract and contributions. The discussion states that pooling needs a trusted holder (building gateway, "
              r"energy community, supplier) and that only one home per client applies otherwise.")


def plain(text: str, macros: dict) -> str:
    t = expand(text, macros)
    t = t.replace("%CLIENTONLY_RESPONSE%", "")
    for a, b in [(r"\,", " "), (r"\%", "%"), ("``", "“"), ("''", "”"), ("--", "–"),
                 (r"\times", "×"), (r"\mathrm{JS}", "JS"), (r"\mathrm{CV}", "CV"), (r"\mathrm{Holm}", "Holm"),
                 (r"\varepsilon", "ε"), (r"\alpha", "α"), (r"\mu", "μ"), (r"\rho", "ρ"),
                 (r"\ln", "ln"), (r"\ge", "≥"), (r"\in", "∈"), ("10^{-3}", "10⁻³")]:
        t = t.replace(a, b)
    t = re.sub(r"_\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\\{|\\\}", lambda m: m.group(0)[1], t)
    t = t.replace("$", "")
    return t


def main() -> Path:
    macros = load_macros()
    rows = [(c, CLIENTONLY if r == "%CLIENTONLY_RESPONSE%" else r, w) for c, r, w in ROWS]
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(10.5)
    h = doc.add_heading("Response to the internal review", level=1)
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
    left = [i for i, (_, r, _) in enumerate(rows, 1) if "n/a" in plain(r, macros)]
    print(f"wrote {OUT} ({len(rows)} comments){'; n/a in ' + str(left) if left else ''}")
    return OUT


if __name__ == "__main__":
    main()
