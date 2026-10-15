"""Build the Word version of the manuscript in the IMRaD journal template.

    .venv/bin/python paper/docx/build_docx.py          -> paper/docx/manuscript.docx (about 12 pages)
    .venv/bin/python paper/docx/build_docx.py --long   -> paper/docx/manuscript_long.docx (full length)
Each is also copied to submission/.

The template (paper/docx/template.docx) supplies the page set-up, styles, header, footer and bullet numbering.
Its body is replaced: every paragraph, table and caption is written with the run formatting of the template's own
prototype paragraphs (title, author line, headings, body text, captions, data tables, reference list). The text
comes from content.tex; every number comes from ../generated/numbers.tex and ../generated/table_*.tex. Equations are
Word-native (OMML), converted LaTeX -> MathML (latex2mathml) -> OMML (Word's own mathml2omml.xsl).
"""
from __future__ import annotations

import copy
import re
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.shared import Inches
from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
TEMPLATE = HERE / "template.docx"
VARIANTS = {   # flag -> (content file, output, copy in submission/)
    "": ("content.tex", "manuscript.docx", "manuscript_template.docx"),
    "--long": ("content_long.tex", "manuscript_long.docx", "manuscript_template_long.docx"),
    "--supp": ("content_supp.tex", "supplementary.docx", "supplementary_material.docx"),
}
FLAG = next((f for f in VARIANTS if f and f in sys.argv), "")
SUPP = FLAG == "--supp"
CONTENT, OUT, COPY_TO = HERE / VARIANTS[FLAG][0], HERE / VARIANTS[FLAG][1], ROOT / "submission" / VARIANTS[FLAG][2]
XSL_CANDIDATES = [HERE / "mathml2omml.xsl",
                  Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl")]

TITLE = "Fragmentation, not heterogeneity, drives the accuracy cost of federated household load forecasting"
AUTHORS = [("Pushkal Gupta", "1"), ("Vanka Kaushik", "1"), ("Kalaavathi B", "1,*")]
AFFILIATIONS = [("1", "School of Computer Science and Engineering, Vellore "
                      "Institute of Technology, Vellore 632014, Tamil Nadu, India")]
CORRESPONDING = "* Corresponding author: Kalaavathi B, kalaavathi.b@vit.ac.in"
EMAILS = "E-mail addresses: pushkal.gupta2023@vitstudent.ac.in (P. Gupta), vanka.kaushik2023@vitstudent.ac.in (V. Kaushik)"
ORCID = ("ORCID: 0009-0004-7032-6847 (P. Gupta), 0009-0003-0735-264X (V. Kaushik),\n"
         "0000-0003-4027-3137 (Kalaavathi B)")
KEYWORDS = ["federated learning", "load forecasting", "smart meters", "non-IID data", "communication efficiency",
            "personalisation", "Low Carbon London"]
HEADER = "Manuscript for review"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = f'xmlns:w="{W}" xmlns:m="{M}"'
TEXT_WIDTH = 9026                      # twips, A4 with 1-inch margins (from the template's sectPr)

# --------------------------------------------------------------------------------------------- macros and text


def balanced(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{'; return the content up to the matching brace and the index after it."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced braces near: {s[i:i + 60]!r}")


def load_macros() -> dict[str, str]:
    src = "\n".join((PAPER / "generated" / f).read_text() for f in ("numbers.tex", "numbers_revision.tex")
                    if (PAPER / "generated" / f).exists())
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", src):
        out[m.group(1)], _ = balanced(src, m.end())
    return out


def expand(text: str, macros: dict[str, str]) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)                               # comments
    text = text.replace("MAPE$^\\ast$", "MAPE*")
    for _ in range(3):
        text = re.sub(r"\\([A-Za-z]+)(\{\})?",
                      lambda m: macros[m.group(1)] if m.group(1) in macros else m.group(0), text)
    text = re.sub(r"\\nosign\{([^{}]*)\}", lambda m: m.group(1).lstrip("+"), text)
    while "\\pv{" in text:                                              # p-values: "= 0.0070" or "< 10^-4"
        i = text.index("\\pv{")
        body, j = balanced(text, i + 3)
        v = body.strip()
        v = v if v.startswith(("$<", "$=", "$>", "n/a")) else f"= {v}"
        text = text[:i] + v + text[j:]
    text = re.sub(r"(?<![\w.)\]}{^_\-])-(?=\d)", "\u2212", text)        # negative numbers: true minus sign
    text = re.sub(r"\\the\\numexpr\s*(\d+)\s*-\s*(\d+)\s*\\relax", lambda m: str(int(m.group(1)) - int(m.group(2))),
                  text)
    return text


# ------------------------------------------------------------------------------------------------ math -> OMML

_xslt = None


def omml(latex: str, size: int | None = None) -> str:
    global _xslt
    if _xslt is None:
        xsl = next(p for p in XSL_CANDIDATES if p.exists())
        _xslt = etree.XSLT(etree.parse(str(xsl)))
    latex = latex.replace(r"\!", "").replace(r"\,", r"\;").replace(r"\Vert", r"\|").strip()
    tree = _xslt(etree.fromstring(latex_to_mathml(latex).encode()))
    xml = etree.tostring(tree.getroot(), encoding="unicode")
    xml = re.sub(r"^<m:oMath[^>]*>", "<m:oMath>", xml)
    rpr = f'<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>' \
          f'{f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>" if size else ""}</w:rPr>'
    xml = re.sub(r"<m:r>(<m:rPr>.*?</m:rPr>)?", lambda m: f"<m:r>{m.group(1) or ''}{rpr}", xml)
    return xml


# ------------------------------------------------------------------------------------------ inline LaTeX -> runs

GREEK = {"alpha": "α", "beta": "β", "mu": "μ", "rho": "ρ", "varepsilon": "ε", "epsilon": "ε", "Delta": "Δ",
         "pi": "π", "theta": "θ", "infty": "∞", "pm": "±", "times": "×", "in": "∈", "ast": "*", "leq": "≤",
         "geq": "≥", "le": "≤", "ge": "≥", "cdot": "·", "sim": "∼", "ldots": "…"}


def math_to_runs(s: str) -> list[tuple[str, dict]]:
    """Simple math (used inside tables) as Unicode runs with italic, sub- and superscripts."""
    s = s.replace(r"\mathrm", "\\rm").replace(r"\bar\rho", "ρ̄").replace(r"\bar{\rho}", "ρ̄")
    out: list[tuple[str, dict]] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:
                i += 1
                continue
            name = m.group(1)
            i += m.end()
            if name == "rm" and i < len(s) and s[i] == "{":
                body, i = balanced(s, i)
                out.append((body, {}))
            elif name in GREEK:
                out.append((GREEK[name], {"i": name in ("alpha", "mu", "rho", "varepsilon", "epsilon")}))
        elif c in "_^":
            i += 1
            if s[i] == "{":
                body, i = balanced(s, i)
            elif s[i] == "\\":
                m = re.match(r"\\[A-Za-z]+", s[i:])
                body, i = m.group(0), i + m.end()
            else:
                body, i = s[i], i + 1
            body = re.sub(r"\\rm\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2), body)
            body = body.replace("\\ast", "*").replace("-", "−")
            out.append((body, {"sub" if c == "_" else "sup": True}))
        elif c.isalpha():
            out.append((c, {"i": True}))
            i += 1
        else:
            out.append(({"-": "−"}.get(c, c), {}))
            i += 1
    return out


class Refs:
    def __init__(self):
        self.cites: dict[str, int] = {}
        self.labels: dict[str, str] = {}

    def cite(self, keys: str) -> str:
        nums = []
        for k in [k.strip() for k in keys.split(",")]:
            if k not in self.cites:
                self.cites[k] = len(self.cites) + 1
            nums.append(self.cites[k])
        nums = sorted(set(nums))
        groups, start = [], nums[0]
        for a, b in zip(nums, nums[1:] + [None]):
            if b != a + 1:
                groups.append((start, a))
                start = b
        parts = [f"[{a}]–[{b}]" if b - a >= 2 else (f"[{a}], [{b}]" if b > a else f"[{a}]") for a, b in groups]
        return ", ".join(parts)


def runs_xml(text: str, refs: Refs, size: int = 22, base: dict | None = None, math_mode: str = "omml") -> str:
    """Render inline LaTeX to <w:r> / <m:oMath> XML."""
    base = base or {}
    out: list[str] = []

    def run(t: str, fmt: dict):
        if not t:
            return
        f = {**base, **fmt}
        rpr = ""
        if f.get("font"):
            rpr += f'<w:rFonts w:ascii="{f["font"]}" w:hAnsi="{f["font"]}"/>'
        if f.get("b"):
            rpr += "<w:b/><w:bCs/>"
        if f.get("i"):
            rpr += "<w:i/><w:iCs/>"
        if f.get("color"):
            rpr += f'<w:color w:val="{f["color"]}"/>'
        rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        if f.get("sub"):
            rpr += '<w:vertAlign w:val="subscript"/>'
        if f.get("sup"):
            rpr += '<w:vertAlign w:val="superscript"/>'
        out.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r>')

    def plain(t: str) -> str:
        t = t.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”").replace("`", "‘")
        t = t.replace("~", "\u00a0").replace("\\,", "\u202f").replace("\\%", "%").replace("\\&", "&")
        t = t.replace("{,}", ",").replace("\\ldots", "…").replace("\\ ", " ")
        return re.sub(r"\s+", " ", t)

    def walk(s: str, fmt: dict):
        i, buf = 0, ""
        while i < len(s):
            c = s[i]
            if c == "$":
                run(plain(buf), fmt)
                buf = ""
                j = s.index("$", i + 1)
                body = s[i + 1:j]
                if math_mode == "omml":
                    out.append(omml(body, size))
                else:
                    for t, f in math_to_runs(body):
                        run(t, {**fmt, **f})
                i = j + 1
                continue
            if c == "\\" and i + 1 < len(s) and s[i + 1].isalpha():
                m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
                name = m.group(1)
                j = i + m.end()
                handlers = {"emph": {"i": True}, "textit": {"i": True}, "textbf": {"b": True},
                            "texttt": {"font": "Courier New"}}
                if name in handlers and j < len(s) and s[j] == "{":
                    run(plain(buf), fmt)
                    buf = ""
                    body, i = balanced(s, j)
                    walk(body, {**fmt, **handlers[name]})
                    continue
                if name in ("cite", "ref", "eqref", "url") and j < len(s) and s[j] == "{":
                    body, i = balanced(s, j)
                    if name == "cite":
                        buf += refs.cite(body)
                    elif name == "ref":
                        buf += refs.labels.get(body, "??")
                    elif name == "eqref":
                        buf += f"({refs.labels.get(body, '??')})"
                    else:
                        buf += body
                    continue
                if name in ("ldots",):
                    buf += "…"
                    i = j
                    continue
                buf += s[i:j]
                i = j
                continue
            buf += c
            i += 1
        run(plain(buf), fmt)

    walk(text, {})
    return "".join(out)


# ------------------------------------------------------------------------------------------ paragraph builders

def para(inner: str, ppr: str) -> str:
    return f"<w:p {NS}><w:pPr>{ppr}</w:pPr>{inner}</w:p>"


BODY_PPR = '<w:spacing w:after="120" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>'


def body_par(text, refs):
    return para(runs_xml(text, refs), BODY_PPR)


def heading(text, level):
    if level == 1:
        return para(f'<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1F3864"/><w:sz w:val="26"/><w:szCs w:val="26"/>'
                    f'</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>',
                    '<w:pStyle w:val="2"/><w:keepNext/><w:spacing w:before="320" w:after="120"/>')
    return para(f'<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="2F4F7F"/><w:sz w:val="23"/><w:szCs w:val="23"/>'
                f'</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>',
                '<w:pStyle w:val="3"/><w:keepNext/><w:spacing w:before="220" w:after="100"/>')


def bullet(text, refs):
    return para(runs_xml(text, refs),
                '<w:pStyle w:val="16"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
                '<w:spacing w:after="60" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>')


def caption(kind, num, text, refs, keep_next=False):
    inner = (f'<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
             f'<w:t>{kind} {num}.</w:t></w:r>'
             f'<w:r><w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr><w:t xml:space="preserve"> </w:t></w:r>'
             + runs_xml(text, refs, size=19))
    keep = "<w:keepNext/><w:keepLines/>" if keep_next else "<w:keepLines/>"
    return para(inner, f'{keep}<w:spacing w:before="80" w:after="160"/><w:jc w:val="both"/>')


def equation(latex, num):
    return para(f'<w:r><w:tab/></w:r>{omml(latex, 22)}<w:r><w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
                f'<w:tab/><w:t>({num})</w:t></w:r>',
                '<w:tabs><w:tab w:val="center" w:pos="4513"/><w:tab w:val="right" w:pos="9026"/></w:tabs>'
                '<w:spacing w:before="120" w:after="120"/><w:jc w:val="left"/>')


def lead_par(lead, text, refs, align="both"):
    return para(f'<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
                f'<w:t xml:space="preserve">{escape(lead)} </w:t></w:r>' + runs_xml(text, refs),
                f'<w:spacing w:after="80" w:line="276" w:lineRule="auto"/><w:jc w:val="{align}"/>')


# ------------------------------------------------------------------------------------------------------- tables

def parse_tabular(src: str) -> tuple[str, list[list[str]]]:
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", src, re.S)
    spec, body = m.group(1), m.group(2)
    body = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", body)
    rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    minus = lambda c: re.sub(r"(?<![\w.)\]}{^_\-])-(?=\d)", "\u2212", c)
    return spec, [[minus(c.strip()) for c in re.split(r"(?<!\\)&", r)] for r in rows]


def table_xml(spec: str, rows: list[list[str]], refs: Refs, prose: bool = False) -> str:
    ncol = len(rows[0])
    aligns = [a for a in spec if a in "lrc"]
    size = 19 if ncol <= 4 else (17 if ncol <= 6 else 16)

    def vis_len(c):
        return len(re.sub(r"\\[A-Za-z]+|[${}]", "", c))

    def word_len(c):                                        # never split a word across lines
        return max((len(w) for w in re.sub(r"\\[A-Za-z]+|[${}]", "", c).split()), default=0)

    weights = [max(7, max(word_len(r[j]) for r in rows) + 3, min(30, max(vis_len(r[j]) for r in rows) + 2))
               for j in range(ncol)]
    if prose:                                               # Table 1 (related work): even widths
        weights = [1.0, 1.25, 1.1, 1.15]
    tot = sum(weights)
    widths = [int(TEXT_WIDTH * w / tot) for w in weights]
    widths[-1] += TEXT_WIDTH - sum(widths)
    if not prose:                                           # room for the longest word of each column
        need = [int(max(word_len(r[j]) for r in rows) * size * 5.5) + 200 for j in range(ncol)]
        short = sum(max(0, n - w) for n, w in zip(need, widths))
        slack = [max(0, w - n) for n, w in zip(need, widths)]
        if short and sum(slack) > short:
            widths = [max(w, n) if n > w else w - int(short * s / sum(slack))
                      for n, w, s in zip(need, widths, slack)]
            widths[-1] += TEXT_WIDTH - sum(widths)
    border = ''.join(f'<w:{s} w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
                     for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    xml = [f'<w:tbl {NS}><w:tblPr><w:tblStyle w:val="9"/><w:tblW w:w="{TEXT_WIDTH}" w:type="dxa"/>'
           f'<w:jc w:val="center"/><w:tblBorders>{border}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="10" w:type="dxa"/>'
           f'<w:bottom w:w="0" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"]
    for r, row in enumerate(rows):
        head = r == 0
        xml.append("<w:tr>" + ("<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if head else
                               "<w:trPr><w:cantSplit/></w:trPr>"))
        for j, cell in enumerate(row):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="1F3864"/>' if head else ""
            jc = {"l": "left", "r": "right", "c": "center"}[aligns[j] if j < len(aligns) and not head else "l"]
            if head and j < len(aligns) and aligns[j] == "r":
                jc = "right"
            fmt = {"b": True, "color": "FFFFFF"} if head else {}
            if not head and row[0].startswith("This work"):
                fmt = {"b": True}
            if re.fullmatch(r"-\d[\d.,]*( \(.*\))?|-\d.*%?", cell.strip()):
                cell = "−" + cell.strip()[1:]                              # true minus sign
            cell = re.sub(r"(?<![\w$])-(?=\d)", "−", cell)
            inner = runs_xml(cell, refs, size=size, base=fmt, math_mode="text")
            xml.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{shade}<w:tcMar>'
                       f'<w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
                       f'<w:right w:w="80" w:type="dxa"/></w:tcMar></w:tcPr>'
                       f'<w:p><w:pPr>{"<w:keepNext/>" if r < len(rows) - 1 else ""}'
                       f'<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
                       f'<w:jc w:val="{jc}"/></w:pPr>{inner}</w:p></w:tc>')
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


# --------------------------------------------------------------------------------------------------- references

ACCENTS = {r"\'a": "á", r"\'e": "é", r"\'i": "í", r"\'o": "ó", r"\'u": "ú", r"\'y": "ý", r'\"u': "ü", r'\"o': "ö",
           r'\"i': "ï", r'\"a': "ä", r"\~a": "ã", r"\~n": "ñ", r"\c{c}": "ç", r"\'{i}": "í", r"\'{\i}": "í",
           r"\v{c}": "č", r"\v{s}": "š", r"\v{z}": "ž", r"\vc": "č", r"\vs": "š"}


def detex(s: str) -> str:
    s = re.sub(r"\{\\v\{?([a-z])\}?\}", lambda m: {"c": "č", "s": "š", "z": "ž"}.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\{\\([\"'~`^])\{?\\?([A-Za-z])\}?\}", lambda m: ACCENTS.get(f"\\{m.group(1)}{m.group(2)}", m.group(2)), s)
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = s.replace("\\&", "&").replace("---", "—").replace("--", "–").replace("\\_", "_")
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bib(path: Path) -> dict[str, dict]:
    src = path.read_text()
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", src):
        kind, key = m.group(1).lower(), m.group(2)
        body, _ = balanced(src, src.index("{", m.start()))
        fields = {"_type": kind}
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            name = f.group(1).lower()
            j = f.end()
            if j < len(body) and body[j] == "{":
                val, _ = balanced(body, j)
            else:
                val = re.match(r"[^,\n]*", body[j:]).group(0)
            fields.setdefault(name, val)
        out[key] = fields
    return out


KEEP = {"London", "Low", "Carbon", "Jensen", "Shannon", "Dirichlet", "Internet", "Things",
        "Ontario", "Canada", "Edmonton", "Texas", "Pecan", "Street", "Holm", "Wilcoxon", "Bayesian", "Kullback",
        "Leibler", "Markov", "Newton", "Adam", "Flower", "Mbed", "Arduino", "UK", "I", "Moreau", "Gaussian"}


def sentence_case(title: str) -> str:
    """Sentence case for reference titles. Brace-protected text, acronyms, mixed-case words and proper nouns are
    kept; the first word and the first word after a colon or dash keep their capital."""
    out, i, start = [], 0, True
    for m in re.finditer(r"\{[^{}]*\}|[^\s{}]+|\s+", title):
        tok = m.group(0)
        if tok.isspace():
            out.append(tok)
            continue
        if tok.startswith("{") and "\\" not in tok:
            out.append(tok)
            start = tok.rstrip("}").endswith((":", "?", "—"))
            continue
        pieces = re.split(r"(—|-)", tok)
        for j in range(0, len(pieces), 2):
            w = pieces[j]
            core = re.sub(r"[^A-Za-z]", "", w)
            simple = core[:1].isupper() and core[1:].islower()
            first_of_title = start and j == 0
            if simple and not first_of_title and core not in KEEP and "\\" not in w:
                pieces[j] = w.lower()
        tok = "".join(pieces)
        out.append(tok)
        start = tok.endswith((":", "?")) or tok in ("--", "---", "—")
    return "".join(out)


def fmt_name(n: str) -> str:
    n = n.strip()
    if n.startswith("{") and n.endswith("}"):
        return detex(n)
    n = detex(n)
    if "," in n:
        last, first = [x.strip() for x in n.split(",", 1)]
    else:
        parts = n.split()
        last, first = parts[-1], " ".join(parts[:-1])
    initials = "".join("-".join(p[0] + "." for p in w.split("-") if p) for w in first.split() if w)
    return f"{last}, {initials}" if initials else last


def fmt_reference(e: dict) -> str:
    raw = [a for a in re.split(r"\s+and\s+", e.get("author", "")) if a.strip()]
    others = any(a.strip() == "others" for a in raw)
    names = [fmt_name(a) for a in raw if a.strip() != "others"]
    authors = ", ".join(names[:6]) + (", et al." if len(names) > 6 or others else "")
    year = detex(e.get("year", "n.d."))
    title = detex(sentence_case(e.get("title", ""))).rstrip(".")
    pages = detex(e.get("pages", "")).replace("–", "-")
    link = f"https://doi.org/{detex(e['doi'])}" if e.get("doi") else detex(e.get("url", ""))
    t = e["_type"]
    if t == "article":
        vol = detex(e.get("volume", ""))
        num = f"({detex(e['number'])})" if e.get("number") else ""
        where = detex(e.get("journal", ""))
        if vol:
            where += f", {vol}{num}"
        if pages:
            where += f": {pages}"
        s = f"{authors} ({year}). {title}. {where}."
    elif t == "inproceedings":
        book = detex(e.get("booktitle", ""))
        book = book if book.lower().startswith(("proceedings", "advances", "international", "icc", "neurips")) \
            or "workshop" in book.lower() or "(" in book else f"Proceedings of the {book}"
        s = f"{authors} ({year}). {title}. In {book}"
        if e.get("volume"):
            s += f", vol. {detex(e['volume'])}"
        if pages:
            s += f", pp. {pages}"
        s += "."
    elif t == "techreport":
        s = (f"{authors} ({year}). {title}. {detex(e.get('type', 'Report'))} No. {detex(e.get('number', ''))}. "
             f"{detex(e.get('institution', ''))}.")
    elif t == "book":
        s = f"{authors} ({year}). {title}. {detex(e.get('publisher', ''))}."
    else:
        how = detex(e.get("howpublished", ""))
        url = re.search(r"https?://\S+", how)
        if url and not link:
            link = url.group(0)
        how = re.sub(r",?\s*https?://\S+", "", how).rstrip(", ")
        s = f"{authors} ({year}). {title}. {how}." if how else f"{authors} ({year}). {title}."
        if e.get("eprint") and not e.get("doi"):
            s += f" arXiv:{detex(e['eprint'])}."
    if link:
        s += f" {link}"
    if e["_type"] == "misc" and "Accessed" in e.get("note", ""):
        acc = re.search(r"Accessed ([^.]*)", e["note"]).group(1)
        s += f", accessed on {acc}"
    return re.sub(r"([.?!])\.", r"\1", s)


def reference_par(num: int, text: str) -> str:
    return para(f'<w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t>[{num}]</w:t></w:r>'
                f'<w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:tab/></w:r>'
                f'<w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>',
                '<w:spacing w:after="80"/><w:ind w:left="500" w:hanging="500"/><w:jc w:val="left"/>')


# ------------------------------------------------------------------------------------------------------- parser

def blocks(src: str):
    """Yield (kind, payload) blocks from content.tex in document order."""
    pos = 0
    pat = re.compile(r"\\begin\{(abstract|itemize|equation|table|figure|declarations)\}(.*?)\\end\{\1\}"
                     r"|\\(section|subsection)\{([^}]*)\}", re.S)
    for m in pat.finditer(src):
        yield from paragraphs(src[pos:m.start()])
        if m.group(1):
            yield m.group(1), m.group(2)
        else:
            yield m.group(3), m.group(4)
        pos = m.end()
    yield from paragraphs(src[pos:])


def paragraphs(chunk: str):
    for p in re.split(r"\n\s*\n", chunk):
        p = " ".join(p.split())
        if p:
            yield "par", p


def label_of(s: str) -> str | None:
    m = re.search(r"\\label\{([^}]*)\}", s)
    return m.group(1) if m else None


# --------------------------------------------------------------------------------------------------------- main

def build() -> Path:
    if not TEMPLATE.exists():
        sys.exit(f"Template not found: {TEMPLATE}")
    global TITLE
    macros = load_macros()
    src = expand(CONTENT.read_text(), macros)
    tm = re.search(r"\\title\{([^}]*)\}", src)                     # optional \title{} in the content file
    if tm:
        TITLE = " ".join(tm.group(1).split())
        src = src[:tm.start()] + src[tm.end():]
    prefix = "S" if SUPP else ""
    items = list(blocks(src))
    refs = Refs()

    # pass 1: float numbers in order of first reference (or appearance), equation numbers in order of appearance
    order = {"fig": [], "tab": []}
    for m in re.finditer(r"\\ref\{((fig|tab):[^}]*)\}|\\label\{((fig|tab):[^}]*)\}", src):
        lab, kind = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        if lab not in order[kind]:
            order[kind].append(lab)
    for kind in order:
        for n, lab in enumerate(order[kind], 1):
            refs.labels[lab] = f"{prefix}{n}"
    eqn = 0
    for kind, payload in items:
        if kind == "equation":
            eqn += 1
            if label_of(payload):
                refs.labels[label_of(payload)] = f"{prefix}{eqn}"
    doc = Document(str(TEMPLATE))
    body = doc.element.body
    sect = body.sectPr
    for el in list(body):
        if el is not sect:
            body.remove(el)

    def add(xml: str):
        sect.addprevious(etree.fromstring(xml))

    # front matter, in the template's formatting
    add(para(f'<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr><w:t>{escape(TITLE)}</w:t></w:r>',
             '<w:spacing w:before="240" w:after="160"/><w:jc w:val="center"/>'))
    runs = []
    for k, (name, sup) in enumerate(AUTHORS):
        sep = ", " if k else ""
        runs.append(f'<w:r><w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr><w:t xml:space="preserve">{sep}{escape(name)}</w:t></w:r>'
                    f'<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/><w:vertAlign w:val="superscript"/></w:rPr><w:t>{sup}</w:t></w:r>')
    add(para("".join(runs), '<w:spacing w:after="60"/><w:jc w:val="center"/>'))
    for sup, aff in AFFILIATIONS:
        add(para(f'<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/><w:vertAlign w:val="superscript"/></w:rPr><w:t>{sup}</w:t></w:r>'
                 f'<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr><w:t xml:space="preserve"> {escape(aff)}</w:t></w:r>',
                 '<w:spacing w:after="40"/><w:jc w:val="center"/>'))
    for line, after in ((CORRESPONDING, 40), (EMAILS, 40), (ORCID, 240)) if not SUPP else ((CORRESPONDING, 240),):
        text = "</w:t><w:br/><w:t>".join(escape(s) for s in line.split("\n"))
        add(para(f'<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr><w:t>{text}</w:t></w:r>',
                 f'<w:spacing w:after="{after}"/><w:jc w:val="center"/>'))

    for kind, payload in items:
        if kind == "abstract":
            add(para('<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1F3864"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
                     '<w:t>Abstract</w:t></w:r>', '<w:spacing w:before="120" w:after="100"/>'))
            add(body_par(" ".join(payload.split()), refs))
            add(para('<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
                     '<w:t xml:space="preserve">Keywords: </w:t></w:r>'
                     f'<w:r><w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr><w:t>{escape("; ".join(KEYWORDS))}</w:t></w:r>',
                     '<w:spacing w:before="160" w:after="240"/>'))
    sec = sub = 0
    for kind, payload in items:
        if kind == "abstract":
            continue
        if kind == "section":
            sec += 1
            sub = 0
            add(heading(f"{'S' if SUPP else ''}{sec}. {payload}", 1))
        elif kind == "subsection":
            sub += 1
            add(heading(f"{'S' if SUPP else ''}{sec}.{sub} {payload}", 2))
        elif kind == "par":
            m = re.match(r"\\textbf\{([^}]*)\}\s*(.*)", payload)
            if m and m.group(1).endswith("."):
                add(lead_par(m.group(1), m.group(2), refs))
            else:
                add(body_par(payload, refs))
        elif kind == "itemize":
            for it in [i.strip() for i in re.split(r"\\item", payload) if i.strip()]:
                add(bullet(" ".join(it.split()), refs))
        elif kind == "equation":
            lab = label_of(payload)
            latex = re.sub(r"\\label\{[^}]*\}", "", payload).strip().rstrip(",.").strip()
            add(equation(" ".join(latex.split()), refs.labels[lab] if lab else "?"))
        elif kind == "table":
            lab = label_of(payload)
            cap, _ = balanced(payload, payload.index("\\caption") + len("\\caption"))
            inp = re.search(r"\\input\{([^}]*)\}", payload)
            tab_src = (PAPER / (inp.group(1) if inp.group(1).endswith(".tex") else inp.group(1) + ".tex")).read_text() \
                if inp else payload
            spec, rows = parse_tabular(tab_src)
            filt = re.search(r"\\rows\{(\d+)\}\{([^}]*)\}", payload)          # \rows{col}{regex}: keep matching rows
            if filt:
                c, rx = int(filt.group(1)), filt.group(2)
                rows = [rows[0]] + [r for r in rows[1:] if re.fullmatch(rx, r[c])]
            keep = re.search(r"\\cols\{([\d,]+)\}", payload)                 # \cols{0,2,3}: keep these columns
            if keep:
                idx = [int(x) for x in keep.group(1).split(",")]
                aligns = [a for a in spec if a in "lrc"]
                spec, rows = "".join(aligns[i] for i in idx), [[r[i] for i in idx] for r in rows]
            add(caption("Table", refs.labels[lab], " ".join(cap.split()), refs, keep_next=True))
            add(table_xml(spec, rows, refs, prose=lab == "tab:gap"))
            add(para("", '<w:spacing w:after="120"/>'))
        elif kind == "figure":
            lab = label_of(payload)
            cap, _ = balanced(payload, payload.index("\\caption") + len("\\caption"))
            for m in re.finditer(r"\\includegraphics\[width=([\d.]+)\]\{([^}]*)\}", payload):
                p = doc.add_paragraph()
                p.paragraph_format.alignment = 1
                p.paragraph_format.keep_with_next = True
                p.paragraph_format.space_before = 0
                p.add_run().add_picture(str(PAPER / "figures" / m.group(2)), width=Inches(6.27 * float(m.group(1))))
            add(caption("Figure", refs.labels[lab], " ".join(cap.split()), refs))
        elif kind == "declarations":
            add(heading("Declarations", 1))
            for d in [d for d in re.split(r"\n\s*\n", payload) if d.strip()]:
                m = re.match(r"\s*\\lead\{([^}]*)\}\s*(.*)", d, re.S)
                add(lead_par(m.group(1), " ".join(m.group(2).split()), refs, align="left"))

    # references, numbered in order of first citation
    bib = parse_bib(PAPER / "refs.bib")
    if refs.cites:
        add(heading("References", 1))
    missing = [k for k in refs.cites if k not in bib]
    if missing:
        sys.exit(f"Missing bibliography keys: {missing}")
    for key, n in sorted(refs.cites.items(), key=lambda kv: kv[1]):
        add(reference_par(n, fmt_reference(bib[key])))

    # header text and document properties
    for p in doc.sections[0].header.paragraphs:
        for r in p.runs:
            r.text = ""
        if p.runs:
            p.runs[0].text = HEADER
    cp = doc.core_properties
    cp.title, cp.author, cp.last_modified_by = TITLE, "Pushkal Gupta; Vanka Kaushik; Kalaavathi B", "Pushkal Gupta"
    cp.subject, cp.keywords, cp.comments = "Manuscript", "; ".join(KEYWORDS), ""

    doc.save(str(OUT))
    unresolved = "??" in "".join(t.text for t in doc.element.body.iter(f"{{{W}}}t") if t.text)
    COPY_TO.parent.mkdir(exist_ok=True)
    shutil.copy(OUT, COPY_TO)
    print(f"wrote {OUT} ({len(refs.cites)} references, {eqn} equations, "
          f"{len(order['tab'])} tables, {len(order['fig'])} figures)")
    if unresolved:
        print("WARNING: unresolved references")
    return OUT


if __name__ == "__main__":
    build()
