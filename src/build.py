# -*- coding: utf-8 -*-
from kit import *

# ───────────────────────────── 1. TITLE ─────────────────────────────
s = slide(number=False)
bar(s, 0, 0, W, 0.10, ACC)
card(s, 8.55, 1.15, 4.10, 4.60, CARD, BORDER)
tb(s, 0.90, 1.55, 7.40, 1.70,
   "Short-Term Household\nEnergy Load Forecasting", 38, True, TXT, spacing=1.08)
tb(s, 0.92, 3.34, 7.40, 0.50,
   "Using Federated LSTM Across Smart Meters", 20, False, ACC)
bar(s, 0.92, 4.02, 0.80, 0.035, ACC)
tb(s, 0.92, 4.32, 4.00, 0.32, "Pushkal Gupta", 15, True, TXT)
tb(s, 0.92, 4.62, 4.00, 0.28, "23BCT0253", 12.5, False, MUT)
tb(s, 4.60, 4.32, 4.00, 0.32, "Kaushik Vanka", 15, True, TXT)
tb(s, 4.60, 4.62, 4.00, 0.28, "23BCT0136", 12.5, False, MUT)
tb(s, 0.92, 5.35, 7.40, 0.30,
   "BCSE316L — Design of Smart Cities  ·  Review 1", 12, False, DIM)

tb(s, 8.90, 1.52, 3.40, 0.30, "Review 1", 11, True, ACC)
for i, (k, v) in enumerate([("City in focus", "London, UK"),
                            ("Comparison cities", "Amsterdam · Copenhagen"),
                            ("Core method", "Federated LSTM (FedAvg)"),
                            ("Alternative", "FedProx for non-IID clients"),
                            ("References", "27 papers, 2022–2026")]):
    y = 2.42 + i * 0.66
    tb(s, 8.90, y, 3.40, 0.24, k.upper(), 8.5, True, DIM)
    tb(s, 8.90, y + 0.24, 3.40, 0.30, v, 12.5, True, TXT)

# ───────────────────────────── 2. CONTENTS ─────────────────────────────
s = slide()
y = head(s, None, "Contents", y0=1.90)

secs = [
    ("1", "Introduction", ACC,
     ["1.1  Topic and context", "1.2  Strengths and weaknesses of the city",
      "1.3  Aim and objectives", "1.4  Scope and limitations"], 3, 9),
    ("2", "Contextual background", WARN,
     ["2.1  Conceptual framework / research design", "2.2  Proposed idea",
      "2.3  Literature survey (27 articles)", "2.4  Key features, challenges, benefits",
      "2.5  Data collection and organisation"], 10, 24),
]
for i, (num, name, col, subs, a, b) in enumerate(secs):
    x = ML + i * 6.20
    card(s, x, y, 5.83, 3.62)
    bar(s, x, y, 5.83, 0.055, col)
    tb(s, x + 0.35, y + 0.42, 0.7, 0.55, num, 34, True, col)
    tb(s, x + 1.05, y + 0.52, 3.2, 0.36, name, 18, True, TXT)
    tb(s, x + 1.05, y + 0.90, 3.2, 0.26, f"Slides {a}–{b}", 10, False, DIM)
    bar(s, x + 0.35, y + 1.36, 5.13, 0.012, BORDER)
    bullets(s, x + 0.35, y + 1.62, 5.10, 2.5, subs, 12, TXT, col, gap=11)

tb(s, ML, y + 3.92, CW, 0.30,
   "Closing slides — evaluation plan, expected outcomes and references.",
   11, False, DIM, PP_ALIGN.CENTER)

# ───────────────────────────── 3. DIVIDER 1 ─────────────────────────────
s = slide()
divider(s, "1", "Introduction", "Section 1",
        ["Topic and context — what the project forecasts and where",
         "Strengths and weaknesses of the city — London, Amsterdam, Copenhagen",
         "Aim and objectives — one aim, four measurable objectives",
         "Scope and limitations — what a one-semester project can honestly deliver"])

# ───────────────────────────── 4. 1.1 TOPIC ─────────────────────────────
s = slide()
y = head(s, "1.1  Topic and context", "What the project forecasts, and where",
         "A smart-meter data problem, a machine-learning problem and a data-governance problem at the same time.")

card(s, ML, y, 7.55, 2.30)
tb(s, ML + 0.35, y + 0.28, 6.9, 0.30, "THE TOPIC", 10, True, ACC)
tb(s, ML + 0.35, y + 0.62, 6.9, 1.45,
   "Predict the short-term electricity demand of an individual household — the "
   "next half-hour to next 24 hours of consumption — by training a shared LSTM "
   "across many smart meters, so that the raw consumption record never leaves "
   "the home gateway.", 14.5, False, TXT, spacing=1.30)

card(s, 8.50, y, 4.18, 2.30, CARD2, BORDER)
tb(s, 8.85, y + 0.28, 3.5, 0.30, "SDG LINKAGE", 10, True, ACC)
for i, (n, t) in enumerate([("SDG 7", "Affordable & clean energy"),
                            ("SDG 11", "Sustainable cities"),
                            ("SDG 12", "Responsible consumption")]):
    yy = y + 0.66 + i * 0.52
    tb(s, 8.85, yy, 0.85, 0.28, n, 13, True, ACC)
    tb(s, 9.75, yy + 0.02, 2.7, 0.28, t, 11, False, MUT)

items = [
    ("THE SETTING",
     "A dense urban distribution network in which every home has a smart meter. "
     "Each meter is a data owner, not a row in one database."),
    ("WHY IT MATTERS NOW",
     "Half-hourly demand data drives peak-load planning, time-of-use tariffs and "
     "flexibility markets — and EV and heat-pump uptake is sharpening the evening peak."),
    ("THE OBSTACLE",
     "Fine-grained traces reveal occupancy and routine, so consent rules and GDPR "
     "data-minimisation limit how much can be pooled centrally."),
]
for i, (k, v) in enumerate(items):
    x = ML + i * 4.05
    card(s, x, y + 2.52, 3.83, 1.95)
    bar(s, x, y + 2.52, 0.055, 1.95, ACC)
    tb(s, x + 0.30, y + 2.78, 3.3, 0.28, k, 10, True, ACC)
    tb(s, x + 0.30, y + 3.16, 3.3, 1.25, v, 11.5, False, MUT, spacing=1.22)

# ───────────────────────────── 5. 1.1 TENSION ─────────────────────────────
s = slide()
y = head(s, "1.1  Topic and context", "Why this is a smart-city problem",
         "Four forces meet in the same place — and they pull against each other.")

pil = [("ENERGY\nFORECASTING", "Short-term demand forecasts drive peak-load planning, demand-side management and tariff design."),
       ("PRIVACY", "Half-hourly traces expose when a home is occupied, when people sleep, cook and leave."),
       ("HETEROGENEITY", "Homes differ in occupancy, appliances, insulation and routine — profiles are not identically distributed."),
       ("DISTRIBUTED IoT", "Smart meters create thousands of data-owning endpoints rather than one central dataset.")]
for i, (k, v) in enumerate(pil):
    x = ML + i * 3.06
    card(s, x, y, 2.85, 2.85)
    tb(s, x + 0.25, y + 0.35, 2.35, 0.70, k, 14, True, ACC, PP_ALIGN.CENTER, 1.15)
    tb(s, x + 0.25, y + 1.18, 2.35, 1.45, v, 11.5, False, MUT, PP_ALIGN.CENTER, 1.25,
       anchor=MSO_ANCHOR.MIDDLE)

card(s, ML, y + 3.12, CW, 1.52, CARD2, ACC)
tb(s, ML + 0.4, y + 3.38, CW - 0.8, 0.32, "THE CORE TENSION", 10, True, ACC, PP_ALIGN.CENTER)
tb(s, ML + 0.9, y + 3.76, CW - 1.8, 0.80,
   "Forecasting gets better with more households' data — but centralising raw household "
   "data is exactly what privacy rules, consent fatigue and data governance resist.",
   16, True, TXT, PP_ALIGN.CENTER, 1.28)

# ───────────────────────── 6. 1.2 CITIES OVERVIEW ─────────────────────────
s = slide()
y = head(s, "1.2  Strengths and weaknesses of the city",
         "Three candidate cities, one chosen as the focus",
         "London is the focus city because it is the only one of the three that publishes a large, "
         "open, household-level half-hourly dataset. Amsterdam and Copenhagen are kept as comparison cases.")

rows = [
    ["City", "Why it is relevant", "Strengths for this project", "Weaknesses / pain points"],
    ["London\n(focus city)",
     "Low Carbon London published ~5,500 households at half-hourly resolution, "
     "labelled with ACORN socio-economic groups.",
     "Open-data culture (London Datastore); a national smart-metering communications "
     "backbone; an active distribution operator running flexibility and time-of-use trials.",
     "Old, poorly insulated housing stock makes demand volatile; very mixed tenancy produces "
     "strongly non-IID profiles; rollout is still incomplete; consent rules restrict central pooling."],
    ["Amsterdam\n(comparison)",
     "High smart-meter penetration combined with a well-documented distribution-grid "
     "congestion problem.",
     "Amsterdam Smart City living-lab programme; near-universal smart meters from the "
     "national rollout; strong local energy-community activity.",
     "Grid congestion in the region is delaying new connections; households may refuse "
     "fine-grained readings, so centrally held data is often coarse."],
    ["Copenhagen\n(comparison)",
     "A high renewable share makes accurate short-term demand forecasting and flexibility "
     "essential rather than optional.",
     "Nationwide remote-read meters; a central metering DataHub; an explicit carbon-neutrality "
     "programme; extensive district heating.",
     "Wind-driven supply variability; district heating shifts the electricity profile, limiting "
     "transfer of a model trained elsewhere; DataHub access is tightly governed."],
]
table(s, ML, y, CW, rows, [1.55, 2.85, 3.75, 3.88],
      hdr_h=0.38, row_h=1.42, hdr_size=10, body_size=9.5)

tb(s, ML, y + 4.72, CW, 0.30,
   "Chosen focus: London — the weaknesses listed above are precisely the conditions "
   "federated learning is designed for.", 11.5, True, ACC, PP_ALIGN.CENTER)

# ───────────────────────── 7. 1.2 LONDON DEEP DIVE ─────────────────────────
s = slide()
y = head(s, "1.2  Strengths and weaknesses of the city",
         "Focus city: London",
         "Read the right-hand column as the requirements list for the system design.")

card(s, ML, y, 5.83, 3.76)
bar(s, ML, y, 5.83, 0.055, ACC)
tb(s, ML + 0.35, y + 0.34, 5.1, 0.32, "STRENGTHS", 14, True, ACC)
bullets(s, ML + 0.35, y + 0.86, 5.15, 2.5, [
    "Low Carbon London trial: about 5,500 households, half-hourly readings, "
    "published openly with ACORN socio-demographic groups.",
    "A national smart-metering communications backbone plus a mature open-data "
    "portal (London Datastore).",
    "An active distribution network operator running time-of-use and flexibility trials.",
    "Dense, well-instrumented network with established demand-side management programmes.",
], 11.5, TXT, ACC, gap=9)

card(s, 6.85, y, 5.83, 3.76)
bar(s, 6.85, y, 5.83, 0.055, WARN)
tb(s, 7.20, y + 0.34, 5.1, 0.32, "WEAKNESSES", 14, True, WARN)
bullets(s, 7.20, y + 0.86, 5.15, 2.5, [
    "Ageing, poorly insulated housing stock makes demand volatile and weather-driven.",
    "Extremely mixed dwelling types and tenancy produce strongly non-IID consumption profiles.",
    "Rollout is incomplete, and some meters lose smart functionality after a supplier switch.",
    "Consent and data-minimisation rules restrict pooling fine-grained readings in one store.",
    "Rising EV and heat-pump load is sharpening the winter evening peak.",
], 11.5, TXT, WARN, gap=8)

card(s, ML, y + 3.98, CW, 0.90, CARD2, ACC)
tb(s, ML + 0.45, y + 4.18, CW - 0.9, 0.55,
   "Each weakness maps to a design decision:  heterogeneity → FedProx   ·   "
   "consent limits → federation instead of pooling   ·   peak volatility → short-term horizon.",
   14, True, TXT, PP_ALIGN.CENTER, 1.25)

# ───────────────────────── 8. 1.3 AIM AND OBJECTIVES ─────────────────────────
s = slide()
y = head(s, "1.3  Aim and objectives", "One aim, four measurable objectives")

card(s, ML, y, CW, 1.62, CARD2, ACC)
tb(s, ML + 0.38, y + 0.22, 1.5, 0.28, "AIM", 10, True, ACC)
tb(s, ML + 0.38, y + 0.58, CW - 0.76, 0.90,
   "To determine whether an LSTM trained federatively across household smart meters can "
   "forecast short-term electricity demand at an accuracy close to centralised training, "
   "and to quantify what that costs in convergence, communication and robustness to "
   "heterogeneous households.", 13.5, False, TXT, spacing=1.25)

objs = [
    ("01", "Establish the baselines",
     "Build a centralised LSTM on pooled data and a local-only LSTM per household. "
     "Report MAE, RMSE and MAPE for both."),
    ("02", "Build the federated forecaster",
     "Implement a simulated household-client Federated LSTM with FedAvg, in which raw "
     "time-series records never leave the client."),
    ("03", "Stress-test heterogeneity",
     "Partition clients by consumption profile rather than at random, and simulate "
     "unequal data sizes, stragglers and partial participation."),
    ("04", "Compare under one protocol",
     "Evaluate Local, Centralised, FedAvg and FedProx on accuracy, convergence and "
     "communication cost using identical preprocessing and horizon."),
]
for i, (n, t, d) in enumerate(objs):
    x = ML + i * 3.06
    yy = y + 1.92
    card(s, x, yy, 2.85, 3.02)
    tb(s, x + 0.28, yy + 0.28, 1.0, 0.50, n, 26, True, BORDER)
    tb(s, x + 0.28, yy + 0.92, 2.32, 0.60, t, 13.5, True, ACC, spacing=1.12)
    tb(s, x + 0.28, yy + 1.62, 2.32, 1.25, d, 11, False, MUT, spacing=1.22)

tb(s, ML, y + 5.10, CW, 0.30,
   "Each objective is deliverable within the semester and produces a number that can be tabulated.",
   11, False, DIM, PP_ALIGN.CENTER)

# ───────────────────────── 9. 1.4 SCOPE AND LIMITATIONS ─────────────────────────
s = slide()
y = head(s, "1.4  Scope and limitations", "Explicit boundaries make the plan credible",
         "Declaring the boundary now prevents an over-promised project later.")

card(s, ML, y, 5.83, 3.22)
bar(s, ML, y, 5.83, 0.055, ACC)
tb(s, ML + 0.35, y + 0.32, 5.1, 0.30, "IN SCOPE", 13, True, ACC)
bullets(s, ML + 0.35, y + 0.80, 5.15, 2.1, [
    "Public household smart-meter datasets; cleaning, resampling and time-series windowing.",
    "Software-simulated federated clients; a single shared LSTM architecture.",
    "FedAvg as the primary method, FedProx as the non-IID alternative.",
    "MAE, RMSE, MAPE, convergence, communication cost and non-IID robustness.",
], 11.5, TXT, ACC, gap=9)

card(s, 6.85, y, 5.83, 3.22)
bar(s, 6.85, y, 5.83, 0.055, WARN)
tb(s, 7.20, y + 0.32, 5.1, 0.30, "OUT OF SCOPE", 13, True, WARN)
bullets(s, 7.20, y + 0.80, 5.15, 2.1, [
    "Physical smart-meter deployment or any IoT hardware build.",
    "Real-time grid control or direct appliance switching.",
    "Electricity-market trading or tariff optimisation.",
    "Utility-scale production deployment across a live distribution network.",
], 11.5, TXT, WARN, gap=9)

card(s, ML, y + 3.46, CW, 1.36, CARD2, WARN)
tb(s, ML + 0.42, y + 3.66, CW - 0.84, 0.28, "THE LIMITATION WE STATE UP FRONT", 10, True, WARN)
tb(s, ML + 0.42, y + 4.00, CW - 0.84, 0.76,
   "Federated learning gives data locality, not a formal privacy guarantee. Raw readings stay on "
   "the client, but model updates can still leak information — so we claim privacy-aware data "
   "minimisation, and name DP-SGD and secure aggregation as future extensions, not as results.",
   12, False, TXT, spacing=1.26)

# ───────────────────────── 10. DIVIDER 2 ─────────────────────────
s = slide()
divider(s, "2", "Contextual Background", "Section 2",
        ["Conceptual framework and research design",
         "The proposed idea — FedAvg and FedProx over a shared LSTM",
         "Literature survey — 27 articles, 2022–2026, and the gaps they leave",
         "Key features, challenges and benefits to the city",
         "Data collection, databases referred and how the data is organised"])

# ───────────────────────── 11. 2.1 CONCEPTUAL FRAMEWORK ─────────────────────────
s = slide()
y = head(s, "2.1  Conceptual framework", "A three-layer smart-city energy architecture",
         "Data moves upward only as model parameters; intelligence moves downward as a shared model.")

layers = [
    ("LAYER 3", "Smart-city energy intelligence",
     ["Aggregate demand view", "Peak & congestion planning", "Demand-side management", "Time-of-use tariffs"], ACC),
    ("LAYER 2", "Federated learning layer",
     ["Global LSTM model", "Client selection", "FedAvg / FedProx aggregation", "Round scheduling"], ACC),
    ("LAYER 1", "Household / IoT client layer",
     ["Smart meter", "Home gateway", "Local data store", "Local LSTM training"], ACC),
]
BH, GAP = 1.32, 0.44
for i, (tag, name, chips, col) in enumerate(layers):
    yy = y + i * (BH + GAP)
    card(s, ML, yy, 9.55, BH)
    bar(s, ML, yy, 0.055, BH, col)
    tb(s, ML + 0.32, yy + 0.30, 2.5, 0.26, tag, 9.5, True, col)
    tb(s, ML + 0.32, yy + 0.62, 2.55, 0.50, name, 13.5, True, TXT, spacing=1.10)
    for j, c in enumerate(chips):
        cx = 3.55 + j * 1.71
        card(s, cx, yy + 0.38, 1.58, 0.56, CARD2, BORDER, 0.18)
        tb(s, cx + 0.06, yy + 0.44, 1.46, 0.46, c, 9.5, False, MUT,
           PP_ALIGN.CENTER, 1.08, MSO_ANCHOR.MIDDLE)

for i, (up, dn) in enumerate([("↑   forecasts and aggregate demand curves", ""),
                              ("↑   model updates (parameters only)", "↓   global model broadcast")]):
    yy = y + BH + i * (BH + GAP) + 0.06
    tb(s, 3.40, yy, 3.6, 0.30, up, 10.5, True, ACC)
    if dn:
        tb(s, 7.10, yy, 3.1, 0.30, dn, 10.5, True, MUT)

card(s, 10.45, y, 2.23, 3 * BH + 2 * GAP, CARD2, BORDER)
tb(s, 10.72, y + 0.32, 1.75, 0.55, "THE INVARIANT", 10, True, ACC, PP_ALIGN.CENTER, 1.1)
tb(s, 10.72, y + 1.05, 1.75, 1.60,
   "Raw half-hourly\nreadings never\ncross Layer 1.", 15, True, TXT, PP_ALIGN.CENTER, 1.32)
tb(s, 10.72, y + 2.95, 1.75, 1.35,
   "Only weight tensors are transmitted, and only in the upward direction.",
   10.5, False, MUT, PP_ALIGN.CENTER, 1.25)

# ───────────────────────── 12. 2.1 RESEARCH DESIGN ─────────────────────────
s = slide()
y = head(s, "2.1  Research design", "A controlled four-arm comparison",
         "The only way to attribute a difference in error to federation is to hold everything else fixed.")

arms = [("LOCAL-ONLY", "Each household trains alone", "Data: stays home\nCollaboration: none", "Lower bound"),
        ("CENTRALISED", "All data pooled, one model", "Data: leaves home\nCollaboration: total", "Upper bound"),
        ("FedAvg", "Federated, weighted averaging", "Data: stays home\nCollaboration: via weights", "Proposed"),
        ("FedProx", "Federated + proximal term", "Data: stays home\nCollaboration: constrained", "Non-IID variant")]
for i, (n, d, meta, role) in enumerate(arms):
    x = ML + i * 3.06
    col = ACC if i >= 2 else MUT
    card(s, x, y, 2.85, 2.32, CARD if i < 2 else CARD2, BORDER if i < 2 else ACC)
    tb(s, x + 0.25, y + 0.26, 2.35, 0.30, n, 14, True, col)
    tb(s, x + 0.25, y + 0.62, 2.35, 0.48, d, 11, False, TXT, spacing=1.18)
    tb(s, x + 0.25, y + 1.22, 2.35, 0.62, meta, 10, False, MUT, spacing=1.32)
    tb(s, x + 0.25, y + 1.94, 2.35, 0.26, role.upper(), 9, True, DIM)

card(s, ML, y + 2.60, 5.83, 2.10)
tb(s, ML + 0.32, y + 2.84, 5.2, 0.28, "HELD CONSTANT ACROSS ALL FOUR ARMS", 10, True, ACC)
bullets(s, ML + 0.32, y + 3.26, 5.2, 1.3, [
    "LSTM depth, hidden units, optimiser and learning rate",
    "Input window length and forecast horizon",
    "Cleaning, resampling and scaling procedure",
    "Chronological train / validation / test split and random seeds",
], 10.5, MUT, ACC, gap=4)

card(s, 6.85, y + 2.60, 5.83, 2.10)
tb(s, 7.17, y + 2.84, 5.2, 0.28, "DELIBERATELY VARIED", 10, True, WARN)
bullets(s, 7.17, y + 3.26, 5.2, 1.3, [
    "Where the data lives and which aggregation rule is used",
    "Degree of client heterogeneity (random vs profile-based partition)",
    "Fraction of clients participating per round",
    "Number of communication rounds and local epochs",
], 10.5, MUT, WARN, gap=4)

# ───────────────────────── 13. 2.2 PROPOSED IDEA ─────────────────────────
s = slide()
y = head(s, "2.2  Proposed idea", "One federated round, end to end",
         "Each household is a client. The server never sees a kilowatt-hour — only weights.")

steps = [("01", "LOCAL DATA", "Half-hourly readings stay on the home gateway and are windowed locally."),
         ("02", "LOCAL TRAINING", "The client trains the shared LSTM for a few epochs on its own history."),
         ("03", "UPLOAD UPDATE", "Only the updated weight tensors are sent to the server."),
         ("04", "AGGREGATE", "The server merges updates with FedAvg, or FedProx under non-IID clients."),
         ("05", "BROADCAST", "The improved global LSTM is pushed back to every client.")]
BW = (CW - 4 * 0.42) / 5
for i, (n, t, d) in enumerate(steps):
    x = ML + i * (BW + 0.42)
    card(s, x, y, BW, 2.15, CARD2 if i in (2, 3) else CARD, ACC if i in (2, 3) else BORDER)
    tb(s, x + 0.22, y + 0.24, 1.0, 0.28, n, 13, True, ACC)
    tb(s, x + 0.22, y + 0.60, BW - 0.44, 0.30, t, 11.5, True, TXT)
    tb(s, x + 0.22, y + 1.00, BW - 0.44, 1.00, d, 10, False, MUT, spacing=1.25)
    if i < 4:
        tb(s, x + BW + 0.02, y + 0.85, 0.38, 0.40, "→", 22, True, ACC, PP_ALIGN.CENTER)

bar(s, ML + 0.9, y + 2.48, CW - 1.8, 0.02, BORDER)
tb(s, ML, y + 2.58, CW, 0.30,
   "↻   repeat for R communication rounds until the global model converges",
   11.5, True, MUT, PP_ALIGN.CENTER)

card(s, ML, y + 3.12, 5.83, 1.50)
tb(s, ML + 0.35, y + 3.34, 5.1, 0.28, "WHAT MOVES IN THE BASELINE", 10, True, WARN)
tb(s, ML + 0.35, y + 3.70, 5.1, 0.70,
   "Centralised training moves the raw consumption record — every reading, for every home, "
   "into one database.", 12, False, TXT, spacing=1.25)

card(s, 6.85, y + 3.12, 5.83, 1.50, CARD2, ACC)
tb(s, 7.20, y + 3.34, 5.1, 0.28, "WHAT MOVES IN THE PROPOSAL", 10, True, ACC)
tb(s, 7.20, y + 3.70, 5.1, 0.70,
   "Federated training moves a fixed-size weight tensor — independent of how many readings "
   "the household holds.", 12, False, TXT, spacing=1.25)

# ───────────────────────── 14. 2.2 FEDAVG / FEDPROX ─────────────────────────
s = slide()
y = head(s, "2.2  Proposed idea", "The two aggregation rules being compared",
         "Same LSTM, same data pipeline. Only the local objective and the merge rule change.")

card(s, ML, y, 5.83, 3.40)
bar(s, ML, y, 5.83, 0.055, ACC)
tb(s, ML + 0.35, y + 0.32, 5.1, 0.32, "FedAvg  —  baseline federated rule", 14, True, ACC)
card(s, ML + 0.35, y + 0.82, 5.13, 0.62, CARD2, BORDER, 0.15)
tb(s, ML + 0.35, y + 0.82, 5.13, 0.62,
   "w  ←  Σ ( n_k / n ) · w_k", 16, True, TXT, PP_ALIGN.CENTER, 1.0, MSO_ANCHOR.MIDDLE)
bullets(s, ML + 0.35, y + 1.62, 5.15, 1.5, [
    "Each client k returns its locally trained weights w_k.",
    "The server averages them, weighted by the number of samples n_k the client holds.",
    "Simple and communication-cheap — but with non-IID clients the local models drift apart "
    "and their average can be a poor compromise for everybody.",
], 11, MUT, ACC, gap=6)

card(s, 6.85, y, 5.83, 3.40)
bar(s, 6.85, y, 5.83, 0.055, WARN)
tb(s, 7.20, y + 0.32, 5.1, 0.32, "FedProx  —  the non-IID alternative", 14, True, WARN)
card(s, 7.20, y + 0.82, 5.13, 0.62, CARD2, BORDER, 0.15)
tb(s, 7.20, y + 0.82, 5.13, 0.62,
   "min  F_k(w)  +  (μ / 2) · || w − w_global ||²", 16, True, TXT, PP_ALIGN.CENTER, 1.0,
   MSO_ANCHOR.MIDDLE)
bullets(s, 7.20, y + 1.62, 5.15, 1.5, [
    "The round structure is unchanged; only the client's loss gains a proximal term.",
    "That term penalises drifting far from the current global model. Setting μ = 0 recovers FedAvg exactly.",
    "It also tolerates stragglers, because a client that finishes fewer local epochs still "
    "contributes a usable update.",
], 11, MUT, WARN, gap=6)

card(s, ML, y + 3.64, CW, 0.95, CARD2, ACC)
tb(s, ML + 0.45, y + 3.86, CW - 0.9, 0.55,
   "Because μ is the only structural difference, any change in error is attributable to "
   "drift control — which is exactly what makes the comparison controlled.",
   13.5, True, TXT, PP_ALIGN.CENTER, 1.25)

# ───────────────────────── 15. 2.3 SURVEY METHOD + THEMES ─────────────────────────
s = slide()
y = head(s, "2.3  Literature survey", "How the 27 articles were selected",
         "The question paper asks for a minimum of 12 articles; this survey retains 27, all published 2022–2026.")

card(s, ML, y, 4.35, 4.68)
tb(s, ML + 0.32, y + 0.28, 3.7, 0.28, "SEARCH PROTOCOL", 10, True, ACC)
rowsA = [("Databases searched", "Scopus · Web of Science · IEEE Xplore · ScienceDirect · SpringerLink"),
         ("Key terms", "federated learning · short-term load forecasting · smart meter · LSTM · non-IID"),
         ("Window", "2022 – 2026 (100% within five years)"),
         ("Screened", "Title and abstract, then full text"),
         ("Retained", "27 peer-reviewed journal articles")]
for i, (k, v) in enumerate(rowsA):
    yy = y + 0.78 + i * 0.78
    tb(s, ML + 0.32, yy, 3.75, 0.24, k.upper(), 8.5, True, DIM)
    tb(s, ML + 0.32, yy + 0.26, 3.75, 0.48, v, 10.5, False, TXT, spacing=1.20)

themes = [("01", "Forecasting foundations",
           "LSTM, BiLSTM and hybrid deep models establish that household load is a temporal "
           "regression problem with strong daily and weekly structure."),
          ("02", "Federated forecasting works",
           "FedAvg, probabilistic and asynchronous variants show collaborative training without "
           "raw-data pooling is feasible on real meter data."),
          ("03", "Heterogeneity is the hard part",
           "Clustering, personalisation and FedProx are all introduced to cope with non-IID "
           "household and building profiles."),
          ("04", "Privacy and efficiency layers",
           "Differential privacy, secure aggregation, homomorphic encryption, quantisation and "
           "hierarchical FL trade accuracy for protection or bandwidth."),
          ("05", "Where 2025–2026 is heading",
           "The newest work treats client heterogeneity, missing data and robustness as the "
           "central experimental variable rather than an assumption.")]
for i, (n, t, d) in enumerate(themes):
    yy = y + i * 0.95
    card(s, 5.35, yy, 7.33, 0.85)
    tb(s, 5.58, yy + 0.09, 0.5, 0.68, n, 13, True, ACC, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 6.15, yy + 0.09, 2.35, 0.68, t, 11.5, True, TXT, spacing=1.10,
       anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 8.60, yy + 0.09, 3.90, 0.68, d, 9.5, False, MUT, spacing=1.18,
       anchor=MSO_ANCHOR.MIDDLE)

# ───────────────────────── 16–18. 2.3 COMPARISON MATRIX ─────────────────────────
HDR = ["Author (year)", "Method", "Dataset / testbed", "Reported metrics",
       "Limitation — what it leaves open"]
LIT = [
 ("Fernández et al. (2022)", "FL + differential privacy + secure aggregation", "Residential smart meters", "Accuracy vs privacy budget", "Privacy layers added together; cost of each is not separated"),
 ("Fekri et al. (2022)", "Federated RNN; FedAvg and FedSGD", "Residential smart-meter load", "RMSE; local vs central vs FL", "Client heterogeneity is not isolated as a variable"),
 ("Gholizadeh & Musilek (2022)", "FL with hyperparameter-based clustering", "Household electricity", "RMSE; convergence", "Clustering presumes prior knowledge of household profiles"),
 ("Yang et al. (2023)", "Integrated FL with decomposition and clustering", "US and Australian load", "MAPE", "Pipeline complexity hides the contribution of FedAvg itself"),
 ("Liu et al. (2023)", "FedForecast — probabilistic federated forecasting", "Pecan Street", "Point and probabilistic accuracy", "Probabilistic focus; communication budget not central"),
 ("Fekri et al. (2023)", "Asynchronous adaptive federated learning", "Smart-meter load", "Accuracy against rounds", "Confidential dataset limits independent reproduction"),
 ("Qu et al. (2023)", "Personalised FL with differential privacy", "Residential load", "Accuracy; privacy level", "Personalisation makes a plain FedAvg baseline hard to read"),
 ("Wang et al. (2023)", "Adaptive federated learning", "Community buildings", "Accuracy; communication", "Building-level clients, not individual households"),
 ("Khalil et al. (2023)", "Privacy-preserving FL for big-data load", "Building energy", "Forecast error and privacy", "Building-oriented; household non-IID effects not studied"),
 ("Widmer et al. (2023)", "FL compared against model personalisation", "Low Carbon London", "Loss distributions", "Shows benchmark design matters, but no communication budget"),
 ("Dogra et al. (2023)", "Consumer profiling combined with FL", "20 residential buildings", "Accuracy; communication", "Grouping requires profile labels that may not exist"),
 ("Xiao et al. (2023)", "Multi-task selected-shared-private learning", "Multiple buildings", "Prediction accuracy", "No federated privacy mechanism; data still shared"),
 ("Dab et al. (2023)", "Consensus-based time-series clustering", "Residential profiles", "Error and cluster quality", "Decentralised deployment complexity not quantified"),
 ("Sarmento et al. (2024)", "CNN-LSTM-FED on edge devices", "Smart* and BDG2", "Accuracy; edge resource use", "Hybrid architecture confounds the effect of federation"),
 ("Abdulla et al. (2024)", "Adaptive FL with stacked LSTM", "Smart-meter energy", "MAE; overhead; training time", "Not framed as a controlled FedAvg benchmark"),
 ("Singh & Bedi (2024)", "FL + transfer learning + clustering", "~6,000 Australian households", "RMSE; MAE", "Transfer learning and clustering confound plain federation"),
 ("Manzoor et al. (2024)", "Single-layer adaptive aggregation + quantisation", "Two energy datasets", "Communication; storage; loss", "Communication-first; forecasting accuracy is secondary"),
 ("Mao et al. (2024)", "Blockchain-based FL with differential privacy", "Power-load datasets", "Accuracy; poisoning resistance", "Architecture far heavier than a semester prototype"),
 ("Aurangzeb et al. (2024)", "Time2Vec embedding with BiLSTM", "SGSC household data", "MAPE; RMSE; MAE", "Fully centralised; no privacy mechanism at all"),
 ("Huang et al. (2024)", "PCA + back-propagation network with FedAvg", "EASP energy data", "Accuracy; efficiency", "Not an LSTM and not household-level clients"),
 ("Mostafanezhad et al. (2025)", "Personalised FL with poisoning defence", "Residential energy", "Error; compute; robustness", "Security focus sits beyond the core forecasting baseline"),
 ("Kaur et al. (2025)", "LSTM + FedProx + homomorphic encryption + clustering", "Residential energy", "MAE / RMSE; overhead", "Encryption cost masks the isolated contribution of FedProx"),
 ("Shao et al. (2026)", "Personalised FL with graph diffusion", "Chinese power system", "Missing-data robustness", "Adds a graph-learning layer on top of federation"),
 ("Das et al. (2026)", "Heterogeneity-aware personalised FL", "Smart*, CU-BEMS, UCI, AMPds", "MAE, R²; 1/12/24-step", "Closest benchmark to ours, but far more complex per client"),
 ("Wang et al. (2026)", "Personalised FL + mixture of experts + conformal prediction", "Two household datasets", "Accuracy; interval reliability", "Model complexity is high relative to the gain reported"),
 ("Tibermacine et al. (2026)", "GRU with FedAvg / FedProx / FedTrimmedAvg", "Moroccan smart grid", "MAE / RMSE / MAPE; comms", "GRU and zone-level clients rather than individual homes"),
 ("He et al. (2026)", "Hierarchical end–edge–cloud federated learning", "Residential smart meters", "Accuracy; latency; privacy", "Requires edge infrastructure not available in simulation"),
]
CHUNKS = [(0, 9), (9, 18), (18, 27)]
for ci, (a, b) in enumerate(CHUNKS):
    s = slide()
    y = head(s, "2.3  Literature survey",
             f"Comparison matrix  —  part {ci + 1} of 3")
    table(s, ML, y, CW, [HDR] + [list(r) for r in LIT[a:b]],
          [1.95, 2.72, 2.22, 1.92, 3.22],
          hdr_h=0.38, row_h=0.505, hdr_size=9.5, body_size=8.5)
    tb(s, ML, y + 5.10, CW, 0.28,
       f"Articles {a + 1}–{b} of 27  ·  every entry is a peer-reviewed journal article published "
       f"between 2022 and 2026  ·  full bibliographic details on the reference slides.",
       9.5, False, DIM)

# ───────────────────────── 19. 2.3 RESEARCH GAPS ─────────────────────────
s = slide()
y = head(s, "2.3  Research gaps", "What the 27 papers leave unanswered",
         "Each gap below is traced to the rows of the comparison matrix, not asserted in the abstract.")

gaps = [("GAP 1", "Heterogeneity is described, not measured",
         "Many papers acknowledge non-IID households and then fix the partition. None sweeps the "
         "degree of heterogeneity as a controlled variable.", "Rows 2, 3, 7, 11, 16"),
        ("GAP 2", "No like-for-like four-arm comparison",
         "Local-only, centralised and federated results are usually reported under different "
         "architectures or preprocessing, so the gap cannot be attributed to federation.", "Rows 2, 10, 14, 24"),
        ("GAP 3", "Accuracy is rarely priced in bandwidth",
         "Communication cost is either the whole focus or absent. Few papers plot forecasting "
         "error against rounds and update bytes on the same axes.", "Rows 5, 17, 20"),
        ("GAP 4", "Stragglers are assumed away",
         "Simulations typically assume every selected client returns on time with equal data. "
         "Unequal sizes and intermittent participation are seldom tested.", "Rows 6, 15, 27"),
        ("GAP 5", "Federation is treated as privacy",
         "Several papers equate keeping data local with privacy. The papers that add DP or "
         "encryption bundle several mechanisms, so no single cost is isolated.", "Rows 1, 7, 18, 22")]
for i, (n, t, d, tr) in enumerate(gaps):
    x = ML + (i % 3) * 4.05
    yy = y + (i // 3) * 2.28
    card(s, x, yy, 3.83, 2.10)
    tb(s, x + 0.28, yy + 0.22, 1.2, 0.26, n, 9.5, True, WARN)
    tb(s, x + 0.28, yy + 0.54, 3.3, 0.52, t, 12.5, True, TXT, spacing=1.12)
    tb(s, x + 0.28, yy + 1.14, 3.3, 0.72, d, 9.5, False, MUT, spacing=1.20)
    tb(s, x + 0.28, yy + 1.82, 3.3, 0.24, "TRACES TO  " + tr, 8, True, DIM)

card(s, ML + 2 * 4.05, y + 2.28, 3.83, 2.10, CARD2, ACC)
tb(s, ML + 2 * 4.05 + 0.28, y + 2.50, 3.3, 0.26, "OUR POSITION", 9.5, True, ACC)
tb(s, ML + 2 * 4.05 + 0.28, y + 2.82, 3.3, 1.40,
   "We do not claim federated learning is new. We claim that a controlled, "
   "heterogeneity-swept, communication-priced comparison of Local, Centralised, FedAvg and "
   "FedProx on one LSTM is missing — and that is what we build.",
   11, False, TXT, spacing=1.24)

# ───────────────────────── 20. 2.4 KEY FEATURES ─────────────────────────
s = slide()
y = head(s, "2.4  Key features of the project", "Six properties that define the system",
         "Each feature exists to answer one of the five research gaps.")

feats = [("Privacy by data flow",
          "Raw half-hourly readings never leave the home gateway. The design constraint is "
          "architectural, not a policy promise.", "Gap 5"),
         ("One architecture, four arms",
          "Local, Centralised, FedAvg and FedProx all run the same LSTM under the same "
          "preprocessing, so any difference is attributable.", "Gap 2"),
         ("Profile-based partitioning",
          "Clients are split by consumption profile rather than at random, producing "
          "realistic non-IID heterogeneity that can be dialled up or down.", "Gap 1"),
         ("Straggler simulation",
          "Unequal client dataset sizes, slow clients and partial round participation are "
          "modelled explicitly rather than assumed away.", "Gap 4"),
         ("Communication accounting",
          "Rounds × participating clients × update size gives an explicit byte budget plotted "
          "against forecasting error.", "Gap 3"),
         ("Reproducible open stack",
          "Public datasets with PyTorch and Flower, fixed seeds and a scripted pipeline, so "
          "the whole comparison can be re-run.", "All gaps")]
for i, (t, d, g) in enumerate(feats):
    x = ML + (i % 3) * 4.05
    yy = y + (i // 3) * 2.28
    card(s, x, yy, 3.83, 2.10)
    bar(s, x, yy, 0.055, 2.10, ACC)
    tb(s, x + 0.30, yy + 0.26, 3.3, 0.32, t, 13.5, True, ACC)
    tb(s, x + 0.30, yy + 0.70, 3.3, 1.00, d, 10.5, False, MUT, spacing=1.24)
    tb(s, x + 0.30, yy + 1.78, 3.3, 0.24, "ANSWERS  " + g.upper(), 8.5, True, DIM)

# ───────────────────────── 21. 2.4.1 CHALLENGES ─────────────────────────
s = slide()
y = head(s, "2.4.1  Challenges in the project", "What can go wrong, and the planned response",
         "These are stated as risks with mitigations rather than hidden until the results are due.")

rows = [["Challenge", "Why it is hard", "Planned mitigation"],
        ["Non-IID household profiles",
         "Local models drift apart during local epochs, so the FedAvg average can be worse for "
         "every client than its own local model.",
         "Introduce FedProx with a tuned μ; sweep the heterogeneity level and report where FedAvg breaks down."],
        ["Stragglers and dropout",
         "Clients have unequal data volumes and finish at different times; waiting for the slowest "
         "client stalls the round.",
         "Simulate partial participation and variable local epochs; FedProx accepts partial work as a valid update."],
        ["Communication overhead",
         "Every round ships a full weight tensor in both directions; more rounds means better "
         "accuracy but more bandwidth.",
         "Record bytes per round and plot error against cumulative communication; report the knee of the curve."],
        ["Data quality",
         "Meter dropouts, missing intervals, clock and daylight-saving shifts, and long near-zero "
         "periods that destabilise MAPE.",
         "Resample to a fixed grid, impute short gaps, drop unusable clients, and always report MAE and RMSE beside MAPE."],
        ["Simulation fidelity",
         "Software clients have no real bandwidth, battery or CPU limit, so the prototype flatters "
         "real gateway behaviour.",
         "State it as a limitation; report update size and round count so real-world cost can be estimated."],
        ["Compute budget",
         "Many clients across many rounds multiplies training time well beyond a single centralised run.",
         "Fix one primary dataset, cap client count and rounds, and use a second dataset only if time allows."]]
table(s, ML, y, CW, rows, [2.55, 4.45, 5.03],
      hdr_h=0.38, row_h=0.72, hdr_size=10, body_size=9.5, accent=WARN)

# ───────────────────────── 22. 2.4.2 BENEFITS TO THE CITY ─────────────────────────
s = slide()
y = head(s, "2.4.2  Features and benefits to the city", "Who gains, and what they gain",
         "The deliverable is a forecasting capability the city can use without building a central store of household data.")

bens = [("THE NETWORK OPERATOR",
         ["Short-term demand visibility for peak-load and congestion planning.",
          "No custodianship of raw household records, so breach exposure falls.",
          "Scales by adding clients, not by growing one central database."]),
        ("HOUSEHOLDS",
         ["Behavioural data stays on the home gateway.",
          "A new or low-history home still benefits from the city-scale model.",
          "Participation needs a far lighter consent than raw-data sharing."]),
        ("REGULATORS AND THE CITY",
         ["Aligns with data-minimisation expectations by design.",
          "Better forecasts support time-of-use tariffs and flexibility schemes.",
          "A transparent, reproducible method that can be audited."]),
        ("SUSTAINABILITY",
         ["SDG 7 — more efficient demand-side planning.",
          "SDG 11 — data-driven infrastructure for a sustainable city.",
          "SDG 12 — better visibility of household consumption."])]
for i, (t, items) in enumerate(bens):
    x = ML + i * 3.06
    card(s, x, y, 2.85, 3.55)
    bar(s, x, y, 2.85, 0.055, ACC)
    tb(s, x + 0.26, y + 0.34, 2.35, 0.50, t, 11, True, ACC, spacing=1.12)
    bullets(s, x + 0.26, y + 1.02, 2.35, 2.3, items, 10.5, MUT, ACC, gap=9)

card(s, ML, y + 3.80, CW, 0.92, CARD2, ACC)
tb(s, ML + 0.45, y + 4.02, CW - 0.9, 0.55,
   "The city gets the forecasting benefit of pooled data without ever holding the pooled data — "
   "that is the whole proposition.", 14.5, True, TXT, PP_ALIGN.CENTER, 1.25)

# ───────────────────────── 23. 2.5 DATA COLLECTION ─────────────────────────
s = slide()
y = head(s, "2.5  Data collection", "Secondary data sources under consideration",
         "This is a secondary-data study. No human subjects are interviewed or surveyed; "
         "all sources are published research datasets with documented access terms.")

rows = [["Dataset", "Coverage", "Resolution and span", "Role in this project"],
        ["Low Carbon London\n(UK Power Networks)",
         "~5,500 London households, with ACORN socio-economic group labels",
         "Half-hourly kWh,\n2011 – 2014",
         "PRIMARY. Large client population and profile labels give a realistic, "
         "controllable non-IID partition for a London-focused study."],
        ["REFIT\n(Loughborough, UK)",
         "20 UK households, aggregate plus nine appliances per home",
         "8-second,\n2013 – 2015",
         "Cross-check at a much finer resolution; used to test whether conclusions survive resampling."],
        ["UK-DALE",
         "5 UK houses, whole-house plus appliance-level channels",
         "1–6 second appliance,\nfrom 2012",
         "Small-client-count stress case; useful for testing behaviour when a client holds very little data."],
        ["Pecan Street Dataport",
         "US residential homes including solar PV and EV charging",
         "1-minute / 15-minute,\nlicensed access",
         "Optional generalisation check on a different climate, tariff regime and appliance mix."]]
table(s, ML, y, CW, rows, [2.35, 3.05, 2.15, 4.48],
      hdr_h=0.38, row_h=0.98, hdr_size=10, body_size=9.5)

tb(s, ML, y + 4.44, CW, 0.28,
   "One primary dataset is selected for the controlled experiment; the others are used only "
   "as robustness checks if time permits.", 10, False, DIM, PP_ALIGN.CENTER)

# ───────────────────────── 24. 2.5 ORGANISATION ─────────────────────────
s = slide()
y = head(s, "2.5  Data organisation and key findings", "From raw meter readings to federated clients",
         "Every client runs this identical pipeline locally — that is what keeps the comparison controlled.")

pipe = [("01", "INGEST", "Load raw meter files; parse timestamps to a single timezone."),
        ("02", "CLEAN", "Flag dropouts, remove duplicate stamps, drop unusable meters."),
        ("03", "RESAMPLE", "Fix all sources to one common interval; impute short gaps only."),
        ("04", "WINDOW", "Build sliding input/target pairs for the chosen horizon."),
        ("05", "SCALE", "Fit the scaler on the client's own training split only."),
        ("06", "PARTITION", "Assign meters to clients by profile; split chronologically.")]
BW2 = (CW - 5 * 0.16) / 6
for i, (n, t, d) in enumerate(pipe):
    x = ML + i * (BW2 + 0.16)
    card(s, x, y, BW2, 1.62)
    tb(s, x + 0.18, y + 0.20, 0.8, 0.24, n, 10, True, ACC)
    tb(s, x + 0.18, y + 0.50, BW2 - 0.36, 0.26, t, 11, True, TXT)
    tb(s, x + 0.18, y + 0.84, BW2 - 0.36, 0.70, d, 9, False, MUT, spacing=1.20)

card(s, ML, y + 1.92, 5.83, 2.75)
tb(s, ML + 0.32, y + 2.14, 5.2, 0.28, "DATABASES AND REPOSITORIES REFERRED", 10, True, ACC)
tb(s, ML + 0.32, y + 2.54, 5.2, 0.24, "LITERATURE DATABASES", 8.5, True, DIM)
tb(s, ML + 0.32, y + 2.80, 5.2, 0.46,
   "Scopus  ·  Web of Science  ·  IEEE Xplore  ·  ScienceDirect  ·  SpringerLink",
   10.5, False, TXT, spacing=1.20)
tb(s, ML + 0.32, y + 3.36, 5.2, 0.24, "SMART-METER DATA REPOSITORIES", 8.5, True, DIM)
bullets(s, ML + 0.32, y + 3.62, 5.2, 1.0, [
    "London Datastore / UK Data Service — Low Carbon London",
    "UKERC Energy Data Centre — UK-DALE",
    "University of Strathclyde repository — REFIT",
    "Pecan Street Dataport — licensed research access",
], 10, TXT, ACC, gap=3)

card(s, 6.85, y + 1.92, 5.83, 2.75)
tb(s, 7.17, y + 2.14, 5.2, 0.28, "KEY FINDINGS FROM THE DATA REVIEW", 10, True, ACC)
bullets(s, 7.17, y + 2.54, 5.2, 2.0, [
    "ACORN group labels in Low Carbon London let heterogeneity be constructed deliberately "
    "instead of at random — this is why it is the primary source.",
    "Household series are spiky and near zero overnight, so MAPE alone is unsafe; MAE and RMSE "
    "must be reported alongside it.",
    "Trial datasets contain meter dropouts and gaps, so cleaning must happen before windowing.",
    "Source resolutions differ by orders of magnitude, so a common interval is required for "
    "any cross-dataset claim.",
], 10, MUT, ACC, gap=6)

# ───────────────────────── 25. EVALUATION PLAN ─────────────────────────
s = slide()
y = head(s, "Evaluation plan", "Exactly what will be measured",
         "Named metrics, named baselines — so the Review-2 result is a table, not an opinion.")

mets = [("MAE", "(1 / n) · Σ | y − ŷ |",
         "Mean absolute error in kWh. Easy to interpret and robust to the near-zero periods "
         "that are common overnight."),
        ("RMSE", "√( (1 / n) · Σ ( y − ŷ )² )",
         "Penalises large misses more heavily, so it exposes failures at the evening peak — "
         "which is where forecasting errors actually cost money."),
        ("MAPE", "(100 / n) · Σ | ( y − ŷ ) / y |",
         "Scale-free, so households of different sizes can be compared — but it blows up as "
         "consumption approaches zero, so it is never reported alone.")]
for i, (n, f, d) in enumerate(mets):
    x = ML + i * 4.05
    card(s, x, y, 3.83, 2.35)
    tb(s, x + 0.30, y + 0.24, 3.3, 0.32, n, 15, True, ACC)
    card(s, x + 0.30, y + 0.66, 3.25, 0.52, CARD2, BORDER, 0.15)
    tb(s, x + 0.30, y + 0.66, 3.25, 0.52, f, 11.5, True, TXT, PP_ALIGN.CENTER, 1.0,
       MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.30, y + 1.34, 3.3, 0.90, d, 10, False, MUT, spacing=1.24)

sysm = [("COMMUNICATION COST",
         "Rounds × clients per round × update size, reported as total bytes transferred and as "
         "error against cumulative communication."),
        ("CONVERGENCE",
         "Training and validation loss plotted per communication round, for every arm, so the "
         "rate as well as the endpoint is visible."),
        ("NON-IID ROBUSTNESS",
         "Change in MAE, RMSE and MAPE as the heterogeneity of the client partition is increased "
         "from random to strictly profile-based.")]
for i, (t, d) in enumerate(sysm):
    x = ML + i * 4.05
    yy = y + 2.60
    card(s, x, yy, 3.83, 1.85, CARD2, BORDER)
    bar(s, x, yy, 0.055, 1.85, WARN)
    tb(s, x + 0.30, yy + 0.26, 3.3, 0.28, t, 10.5, True, WARN)
    tb(s, x + 0.30, yy + 0.64, 3.3, 1.00, d, 10.5, False, MUT, spacing=1.24)

# ───────────────────────── 26. EXPECTED OUTCOMES ─────────────────────────
s = slide()
y = head(s, "Expected outcomes", "The deliverable is a filled-in comparison table",
         "No accuracy figure is promised in advance — the contribution is the measurement, not a claimed number.")

rows = [["Arm", "Raw data leaves home?", "MAE", "RMSE", "MAPE", "Rounds to converge", "Bytes transferred"],
        ["Local-only LSTM", "No", "—", "—", "—", "n/a", "0"],
        ["Centralised LSTM", "Yes", "—", "—", "—", "n/a", "full dataset"],
        ["FedAvg LSTM", "No", "—", "—", "—", "—", "—"],
        ["FedProx LSTM", "No", "—", "—", "—", "—", "—"]]
table(s, ML, y, CW, rows, [2.60, 2.25, 1.30, 1.30, 1.30, 1.79, 1.79],
      hdr_h=0.42, row_h=0.46, hdr_size=9.5, body_size=10)

tb(s, ML, y + 2.36, CW, 0.26,
   "Every dash is a cell this project exists to fill. The first two columns are already known — "
   "they are the reason the study is worth running.", 10, False, DIM)

outs = [("ARTEFACTS",
         ["Cleaned, windowed household dataset with a documented pipeline.",
          "Local-only and centralised LSTM baselines.",
          "FedAvg federated prototype built on Flower and PyTorch.",
          "FedProx extension with a tuned proximal coefficient."]),
        ("EVIDENCE",
         ["Accuracy table across all four arms under one protocol.",
          "Convergence curves per communication round.",
          "Error-versus-bandwidth curve identifying the useful operating point.",
          "Degradation curve as client heterogeneity increases."]),
        ("THE ARGUMENT",
         ["Where federation gets close to centralised accuracy — and where it does not.",
          "What non-IID households actually cost in kWh of error.",
          "Whether FedProx earns its extra complexity over FedAvg.",
          "A negative result here is still a result worth reporting."])]
for i, (t, items) in enumerate(outs):
    x = ML + i * 4.05
    yy = y + 2.70
    card(s, x, yy, 3.83, 2.28)
    tb(s, x + 0.30, yy + 0.22, 3.3, 0.28, t, 10, True, ACC)
    bullets(s, x + 0.30, yy + 0.58, 3.3, 1.62, items, 9.5, MUT, ACC, gap=4)

# ───────────────────────── 27. CONCLUSION ─────────────────────────
s = slide()
y = head(s, None, "Summary")

pts = [("THE PROBLEM",
        "London needs better short-term household load forecasts, but its housing stock makes "
        "demand heterogeneous and its consent rules make central pooling hard."),
       ("THE PROPOSAL",
        "Treat every household as a federated client. Train one shared LSTM through FedAvg, and "
        "use FedProx where household profiles diverge. Only weights move."),
       ("THE CONTRIBUTION",
        "Not a new architecture — a controlled four-arm comparison that measures what federation "
        "and heterogeneity actually cost in forecasting error and bandwidth."),
       ("THE HONESTY",
        "Federated learning gives data locality, not formal privacy. We will say so, and name "
        "differential privacy and secure aggregation as the next step, not as a result.")]
for i, (t, d) in enumerate(pts):
    x = ML + (i % 2) * 6.20
    yy = y + (i // 2) * 1.62
    card(s, x, yy, 5.83, 1.42)
    bar(s, x, yy, 0.055, 1.42, ACC if i < 3 else WARN)
    tb(s, x + 0.32, yy + 0.24, 5.2, 0.28, t, 10, True, ACC if i < 3 else WARN)
    tb(s, x + 0.32, yy + 0.60, 5.2, 0.70, d, 11.5, False, TXT, spacing=1.24)

card(s, ML, y + 3.42, CW, 1.30, CARD2, ACC)
tb(s, ML + 0.5, y + 3.66, CW - 1.0, 0.85,
   "The city gets a forecasting model trained on thousands of homes,\n"
   "without ever holding the data of a single one of them.",
   18, True, TXT, PP_ALIGN.CENTER, 1.30)

# ───────────────────────── 28–29. REFERENCES ─────────────────────────
REFS = [
 "Fernández, J. D., Menci, S. P., Lee, C. M., Rieger, A., & Fridgen, G. (2022). Privacy-preserving federated learning for residential short-term load forecasting. Applied Energy, 326, 119915.",
 "Fekri, M. N., Grolinger, K., & Mir, S. (2022). Distributed load forecasting using smart meter data: federated learning with recurrent neural networks. Int. J. Electrical Power & Energy Systems, 137, 107669.",
 "Gholizadeh, N., & Musilek, P. (2022). Federated learning with hyperparameter-based clustering for electrical load forecasting. Internet of Things, 17, 100470.",
 "Yang, Y., Wang, Z., Zhao, S., & Wu, J. (2023). An integrated federated learning algorithm for short-term load forecasting. Electric Power Systems Research, 214, 108830.",
 "Liu, Y., Dong, Z., Liu, B., Xu, Y., & Ding, Z. (2023). FedForecast: a federated learning framework for short-term probabilistic individual load forecasting in smart grid. IJEPES, 152, 109172.",
 "Fekri, M. N., Grolinger, K., & Mir, S. (2023). Asynchronous adaptive federated learning for distributed load forecasting with smart meter data. IJEPES, 153, 109285.",
 "Qu, X., Guan, C., Xie, G., Tian, Z., Sood, K., Sun, C., & Cui, L. (2023). Personalized federated learning for heterogeneous residential load forecasting. Big Data Mining and Analytics, 6(4), 421–432.",
 "Wang, R., Yun, H., Rayhana, R., Bin, J., Zhang, C., Herrera, O. E., Liu, Z., & Mérida, W. (2023). An adaptive federated learning system for community building energy load forecasting and anomaly prediction. Energy and Buildings, 295, 113215.",
 "Khalil, M., Esseghir, M., & Merghem Boulahia, L. (2023). Privacy-preserving federated learning: an application for big data load forecast in buildings. Computers & Security, 131, 103211.",
 "Widmer, F., Nowak, S., Bowler, B., Huber, P., & Papaemmanouil, A. (2023). Data-driven comparison of federated learning and model personalization for electric load forecasting. Energy and AI, 14, 100253.",
 "Dogra, S., et al. (2023). Consumers profiling based federated learning approach for energy load forecasting. Sustainable Cities and Society, 98, 104815.",
 "Xiao, J.-W., et al. (2023). Joint load prediction of multiple buildings using multi-task learning with a selected-shared-private mechanism. Energy and Buildings, 293, 113178.",
 "Dab, K., et al. (2023). Consensus-based time-series clustering approach to short-term load forecasting for residential electricity demand. Energy and Buildings, 299, 113550.",
 "de Moraes Sarmento, E. M., et al. (2024). Forecasting energy power consumption using federated learning in edge computing devices. Internet of Things, 25, 101050.",
 "Abdulla, N., Demirci, M., & Ozdemir, S. (2024). Smart meter-based energy consumption forecasting for smart cities using adaptive federated learning. Sustainable Energy, Grids and Networks, 38, 101342.",
 "Singh, G., & Bedi, J. (2024). A federated and transfer learning based approach for households load forecasting. Knowledge-Based Systems, 299, 111967.",
 "Manzoor, H. U., Jafri, A., & Zoha, A. (2024). Adaptive single-layer aggregation framework for energy-efficient and privacy-preserving load forecasting in heterogeneous federated smart grids. Internet of Things, 28, 101376.",
 "Mao, Q., Wang, L., Long, Y., Han, L., Wang, Z., & Chen, K. (2024). A blockchain-based framework for federated learning with privacy preservation in power load forecasting. Knowledge-Based Systems, 284, 111338.",
 "Aurangzeb, K., Haider, S. I., & Alhussein, M. (2024). Individual household load forecasting using bidirectional LSTM network with time-based embedding. Energy Reports, 11, 3963–3975.",
 "Huang, Y., Song, Y., & Jing, Z. (2024). Load forecasting using federated learning with consideration of electricity data privacy preservation of EASP. Ain Shams Engineering Journal, 15, 102724.",
 "Mostafanezhad, M., Mohammadi, M., Keihan Asl, D., & Karimipour, H. (2025). Personalized poison-attack tolerant federated learning for residential load forecasting. Sustainable Energy, Grids and Networks, 43, 101794.",
 "Kaur, I., Bedi, J., & Aggarwal, A. (2025). Privacy preserving and computationally efficient federated learning approach for household load estimation. Sustainable Energy, Grids and Networks, 44, 102021.",
 "Shao, L., Pang, K., Feng, C., Yang, Y., & Wen, F. (2026). Privacy-preserving residential short-term load forecasting with missing data tolerance. IEEE Transactions on Smart Grid, 17(2), 1664–1679.",
 "Das, P., Pandey, S., Pandey, C., & Kumar, D. (2026). Heterogeneity-aware personalised federated learning for household energy forecasting on multi-source real-world smart meter data. Scientific Reports, 16, 24957.",
 "Wang, J., Xu, D., Xing, L., Chen, T., Liu, Y., Shahidehpour, M., & Yang, T. (2026). Personalized federated learning with mixture of experts and conformal prediction for household energy forecasting. Expert Systems with Applications, 300, 130417.",
 "Tibermacine, A., et al. (2026). Privacy-preserving load forecasting in smart grids using federated learning: a comparative analysis of aggregation strategies. Frontiers in Energy Research, 14, 1854654.",
 "He, Z., Zhang, H., Liu, Y., et al. (2026). Privacy-preserving electricity load forecasting using hierarchical federated learning. Journal of Information and Intelligence, 4(4), 350–363.",
]
PAGES = [(0, 14), (14, 27)]
for pi, (a, b) in enumerate(PAGES):
    s = slide()
    y = head(s, "References", f"27 peer-reviewed journal articles, 2022–2026  —  part {pi + 1} of 2")
    chunk = list(range(a, b))
    half = (len(chunk) + 1) // 2
    for ci, col in enumerate([chunk[:half], chunk[half:]]):
        x = ML + ci * 6.20
        for j, ri in enumerate(col):
            yy = y + j * 0.76
            tb(s, x, yy, 0.42, 0.30, f"[{ri + 1}]", 9, True, ACC)
            tb(s, x + 0.45, yy - 0.01, 5.38, 0.70, REFS[ri], 8.5, False, MUT, spacing=1.20)

# ───────────────────────── SAVE ─────────────────────────
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, os.pardir, "deck", "23BCT0253.pptx"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
