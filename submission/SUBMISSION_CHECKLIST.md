# Submission package: Applied Energy (Elsevier)

**Target journal:** *Applied Energy* (Elsevier), research article, via Elsevier Editorial Manager.
**Fallback:** if the editor desk-rejects for scope, transfer to *Energy and AI* (Elsevier), which uses the
same `elsarticle` format and also publishes federated-learning work for energy.

## Files in this folder

| File | Use in Editorial Manager |
|---|---|
| `manuscript.pdf` | Manuscript (initial submission accepts a single PDF) |
| `cover_letter.pdf` | Cover letter |
| `highlights.txt` | Highlights: 5 bullets, each ≤ 85 characters |
| `graphical_abstract.png` | Graphical abstract (3966 × 1665 px; minimum 1328 × 531) |
| `../paper/` | LaTeX source (`manuscript.tex`, `sections/`, `refs.bib`, `generated/`, `figures/`); needed at revision |
| Code/data archive | Supplementary material: zip of the repository without `data/raw` |

## Before you press Submit (only the authors can do these)

- [x] Dr. Kalaavathi B has agreed to be listed as an author (confirmed by the corresponding author, 2026-10-10).
- [ ] **All three authors read and approve** the final manuscript, the author order and the CRediT roles
      (Kalaavathi B: Supervision, Methodology, Writing – review & editing).
- [ ] **Each author revises parts of the text in their own words**, especially the Introduction and Discussion.
      This is the most reliable way to make the paper read as the authors' own work.
- [ ] Confirm the **affiliation** wording for all authors.
- [ ] Confirm the **CRediT statement** in `paper/sections/declarations.tex`.
- [ ] Confirm **funding** ("no specific grant") and **competing interests** ("none").
- [ ] Review the **generative-AI declaration** (Elsevier policy). It states that Claude (Anthropic) was used
      for software and drafting, and that the authors take responsibility.
- [ ] Decide where the code goes. Make the GitHub repo public, or upload it to Zenodo for a DOI, then
      replace the placeholder in the *Data and code availability* section.
- [ ] Read the journal's current Guide for Authors once more. Check word count (about 8,000 for research articles; the body is about 6,000 words now),
      abstract length (198 words) and keywords (6).
- [ ] Run the PDF through your institution's Turnitin/iThenticate **similarity** check. The text is original and
      every borrowed claim is cited, so the score should be low. Remaining matches will mostly be the reference list
      and standard phrases. Keep the generative-AI declaration: Elsevier requires it, and it covers the use of AI tools.
- [ ] Optionally add **suggested reviewers**. Do not suggest anyone with a conflict of interest.
- [ ] Recompile after any edit: `cd paper && tectonic manuscript.tex`, then copy the PDF here.

## Editorial Manager steps

1. Create or log in to an Elsevier account as the corresponding author (pushkal.gupta2023@vitstudent.ac.in).
2. Choose *Research Paper* as the article type.
3. Upload, in order: manuscript PDF, highlights, graphical abstract, cover letter, supplementary archive.
4. Enter the title, abstract and keywords: copy them from the PDF.
5. Add co-authors with their e-mails:
   - Vanka Kaushik: vanka.kaushik2023@vitstudent.ac.in
   - Kalaavathi B: kalaavathi.b@vit.ac.in
6. Complete the declarations (competing interests, funding, data availability, AI use).
7. Review the system-built PDF, then approve and submit.

## Self-review against common desk-rejection reasons

| Risk | Status |
|---|---|
| Incremental ML novelty | The contribution is a controlled decomposition plus negative results, not a new model. It is framed against prior LCL work (Briggs et al. 2022) and adaptive-μ work (FedProx, HAPI-FedProx). |
| Missing energy-system relevance | Deployment implications for DSOs, aggregators and community hubs: client tier, bandwidth, consumption-tier equity. |
| Unsupported claims | Every number in the text is a macro generated from the result files; each claim is tied to a table or test. |
| Single dataset | Stated as a limitation. A second dataset would strengthen a revision. |
| Reproducibility | Code, configs, per-run histories and per-household errors are included. |
