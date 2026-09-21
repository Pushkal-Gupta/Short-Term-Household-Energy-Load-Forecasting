import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

def iter_block(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

d = Document(sys.argv[1])
for b in iter_block(d):
    if isinstance(b, Paragraph):
        t = b.text.strip()
        if t: print(f"[{b.style.name}] {t}")
    else:
        print("--- TABLE ---")
        for r in b.rows:
            print(" | ".join(c.text.strip().replace("\n"," ") for c in r.cells))
        print("--- END TABLE ---")
