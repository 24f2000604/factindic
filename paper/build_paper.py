"""Builds paper/FactIndic_IEEE.docx (python-docx) and FactIndic_IEEE.pdf (headless Chrome) from content.py.
    python paper/compute_numbers.py && python paper/make_figures.py && python paper/build_paper.py"""
import base64
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content as C                                              # noqa: E402
from docx import Document                                        # noqa: E402
from docx.enum.section import WD_SECTION                         # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT                   # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL              # noqa: E402
from docx.oxml import OxmlElement                                # noqa: E402
from docx.oxml.ns import qn                                      # noqa: E402
from docx.shared import Inches, Pt, RGBColor                     # noqa: E402

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
TOK = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*(?!\*)|_\{.+?\}|\^\{.+?\})")
UNNUM = {"Code and Data", "Acknowledgment"}


def spans(t):
    for x in TOK.split(t):
        if not x:
            continue
        if x.startswith("**"):
            yield x[2:-2], "b"
        elif x.startswith("*") and x.endswith("*") and len(x) > 2:
            yield x[1:-1], "i"
        elif x.startswith("_{"):
            yield x[2:-1], "sub"
        elif x.startswith("^{"):
            yield x[2:-1], "sup"
        else:
            yield x, ""


# ------------------------------------------------------------------------------------------------ Word
def build_docx(path):
    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = st.paragraph_format.space_before = Pt(0)
    for n, size in (("Heading 1", 10), ("Heading 2", 10), ("Title", 22)):
        h = d.styles[n]
        h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Times New Roman", Pt(size), False, RGBColor(0, 0, 0)
        rf = h.element.get_or_add_rPr().get_or_add_rFonts()
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(k), "Times New Roman")
        for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            rf.attrib.pop(qn(k), None)
    d.styles["Title"].element.get_or_add_pPr()
    for b in d.styles["Title"].element.pPr.findall(qn("w:pBdr")):
        d.styles["Title"].element.pPr.remove(b)
    d.styles["Heading 1"].font.small_caps = True
    d.styles["Heading 2"].font.italic = True
    sec = d.sections[0]

    def setup(s, cols):
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Inches(.75), Inches(1), Inches(.625), Inches(.625)
        c = s._sectPr.find(qn("w:cols"))
        if c is None:
            c = OxmlElement("w:cols")
            s._sectPr.append(c)
        c.set(qn("w:num"), str(cols)); c.set(qn("w:space"), "360")

    def run(p, t, size=None, **kw):
        for x, k in spans(t):
            r = p.add_run(x)
            r.bold = True if k == "b" else kw.get("bold")
            r.italic = (k == "i") != bool(kw.get("italic")) or None
            r.font.subscript, r.font.superscript = (k == "sub") or None, (k == "sup") or None
            if size:
                r.font.size = Pt(size)

    def para(t, size=None, align=AL.JUSTIFY, indent=True, after=0, before=0, style=None, **kw):
        p = d.add_paragraph(style=style)
        p.alignment = align
        f = p.paragraph_format
        f.first_line_indent = Inches(.14) if indent else None
        f.space_after, f.space_before = Pt(after), Pt(before)
        run(p, t, size, **kw)
        return p

    def borders(cell, **edges):
        pr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        for e in ("top", "bottom"):
            if e in edges:
                x = OxmlElement(f"w:{e}")
                for k, v in (("val", "single"), ("sz", str(edges[e])), ("space", "0"), ("color", "000000")):
                    x.set(qn(f"w:{k}"), v)
                b.append(x)
        pr.append(b)

    setup(sec, 1)
    p = d.add_paragraph(style="Title"); p.alignment = AL.CENTER; p.paragraph_format.space_after = Pt(8)
    run(p, C.TITLE)
    for i, a in enumerate(C.AUTHORS):
        para(a, 11 if i == 0 else 10, AL.CENTER, False, 10 if i == len(C.AUTHORS) - 1 else 0, italic=i == 1)
    setup(d.add_section(WD_SECTION.CONTINUOUS), 2)
    p = para("", 9, indent=True, after=4)
    run(p, "Abstract", 9, bold=True, italic=True); run(p, "—" + C.ABSTRACT, 9, bold=True)
    p = para("", 9, indent=True, after=4)
    run(p, "Index Terms", 9, bold=True, italic=True); run(p, "—" + C.INDEX, 9, bold=True, italic=True)
    n1 = n2 = nf = nt = ne = 0
    for b in C.B:
        k = b[0]
        if k == "h1":
            n2 = 0
            if b[1] in UNNUM:
                t = b[1]
            else:
                n1 += 1; t = f"{ROMAN[n1]}. {b[1]}"
            para(t, None, AL.CENTER, False, 4, 10, "Heading 1").paragraph_format.keep_with_next = True
        elif k == "h2":
            n2 += 1
            para(f"{chr(64 + n2)}. {b[1]}", None, AL.LEFT, False, 3, 6, "Heading 2").paragraph_format.keep_with_next = True
        elif k in ("p", "pc"):
            para(b[1], indent=k == "p")
        elif k == "li":
            p = para(b[1], indent=False, style="List Bullet", after=1)
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(.2), Inches(-.15)
        elif k == "eq":
            ne += 1
            p = para("", indent=False, before=4, after=4, align=AL.LEFT)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(3.45), 2)
            run(p, b[1], 9.5); p.add_run(f"\t({ne})")
        elif k == "fig":
            _, f, cap, wide = b
            nf += 1
            if wide:
                setup(d.add_section(WD_SECTION.CONTINUOUS), 1)
            p = para("", align=AL.CENTER, indent=False, before=6); p.paragraph_format.keep_with_next = True
            pic = p.add_run().add_picture(str(HERE / "figures" / f), width=Inches(7.1 if wide else 3.45))
            pic._inline.docPr.set("descr", cap)
            para(f"Fig. {nf}. {cap}", 8, AL.CENTER if wide else AL.JUSTIFY, False, 8)
            if wide:
                setup(d.add_section(WD_SECTION.CONTINUOUS), 2)
        elif k == "table":
            _, cap, head, rows, wide, note = b
            nt += 1
            if wide:
                setup(d.add_section(WD_SECTION.CONTINUOUS), 1)
            p = para(f"TABLE {ROMAN[nt]}", 8, AL.CENTER, False, 0, 6); p.paragraph_format.keep_with_next = True
            p = para(cap, 8, AL.CENTER, False, 3); p.paragraph_format.keep_with_next = True
            size = 6.5 if wide else 8
            total = 7.2 if wide else 3.45
            first = 1.2 if wide else (1.35 if len(head) == 5 else 1.15 if len(head) == 6 and head[0] == "Method" else .8)
            if wide:
                w = [1.25] + [.4, .36, .36, .36, .36, .36] + [.72] * 3 + [.45, .4, .36, .36, .36]
            else:
                w = [first] + [(total - first) / (len(head) - 1)] * (len(head) - 1)
            tab = d.add_table(rows=1 + len(rows), cols=len(head))
            tab.alignment, tab.autofit = WD_TABLE_ALIGNMENT.CENTER, False
            for i, row in enumerate([head] + rows):
                trp = tab.rows[i]._tr.get_or_add_trPr()
                trp.append(OxmlElement("w:cantSplit"))
                if i == 0:
                    trp.append(OxmlElement("w:tblHeader"))
                for j, x in enumerate(row):
                    c = tab.cell(i, j)
                    c.width = Inches(w[j])
                    cp = c.paragraphs[0]
                    cp.alignment = AL.LEFT if j == 0 else AL.CENTER
                    cp.paragraph_format.keep_with_next = i < len(rows)
                    run(cp, x.replace("*", "∗") if i == 0 else x, size, bold=i == 0)
                    e = {}
                    if i == 0:
                        e = dict(top=8, bottom=4)
                    elif i == len(rows):
                        e = dict(bottom=8)
                    if e:
                        borders(c, **e)
                    m = OxmlElement("w:tcMar")
                    for side in ("left", "right"):
                        y = OxmlElement(f"w:{side}"); y.set(qn("w:w"), "30"); y.set(qn("w:type"), "dxa"); m.append(y)
                    c._tc.get_or_add_tcPr().append(m)
            if note:
                para(note.replace("*", "∗"), 7.5, AL.JUSTIFY, False, 8, 2)
            else:
                para("", 6, indent=False, after=4)
            if wide:
                setup(d.add_section(WD_SECTION.CONTINUOUS), 2)
    para("References", None, AL.CENTER, False, 4, 10, "Heading 1")
    for i, r in enumerate(C.REFS, 1):
        p = para(f"[{i}]\t{r}", 8, AL.JUSTIFY, False, 1)
        p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(.27), Inches(-.27)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(.27))
    z = d.settings.element.find(qn("w:zoom"))           # python-docx's template leaves the required percentage out
    if z is not None:
        z.set(qn("w:percent"), "100")
    d.core_properties.title, d.core_properties.author = C.TITLE, ""
    d.save(path)


# ------------------------------------------------------------------------------------------------ PDF
def H(t):
    tag = dict(b="b", i="i", sub="sub", sup="sup")
    return "".join(f"<{tag[k]}>{html.escape(x)}</{tag[k]}>" if k else html.escape(x) for x, k in spans(t))


CSS = """@page{size:Letter;margin:.75in .625in 1in}*{box-sizing:border-box}
body{font:10pt/1.17 'Times New Roman','Liberation Serif',serif;margin:0;color:#000}
.t{font-size:22pt;text-align:center;line-height:1.12;margin:0 0 8pt;font-weight:400}.a{text-align:center;margin:0 0 10pt;font-size:10pt}
.c{column-count:2;column-gap:.25in;text-align:justify;hyphens:auto}p{margin:0;text-indent:.14in;orphans:2;widows:2}
.n{text-indent:0}.abs{font:bold 9pt/1.17 'Times New Roman','Liberation Serif',serif;margin-bottom:4pt}
h1{font-size:10pt;font-weight:400;font-variant:small-caps;text-align:center;margin:10pt 0 4pt;break-after:avoid}
h2{font-size:10pt;font-weight:400;font-style:italic;text-align:left;margin:6pt 0 3pt;break-after:avoid}
ul{margin:2pt 0;padding-left:.2in}li{margin:0 0 1pt}.eq{display:flex;justify-content:space-between;margin:4pt 0;font-size:9.5pt;text-align:left}
figure{margin:6pt 0 8pt;break-inside:avoid;text-align:center}figure img{width:100%}figcaption{font-size:8pt;text-align:justify;line-height:1.15}
.w{column-span:all}.w figcaption{text-align:center}.tb{break-inside:avoid;margin:6pt 0 8pt;font-size:8pt}.cap{text-align:center;margin-bottom:3pt}
table{border-collapse:collapse;width:100%;border-top:1pt solid #000;border-bottom:1pt solid #000;line-height:1.2}th{border-bottom:.5pt solid #000}
th,td{padding:.6pt 2pt;text-align:center}th:first-child,td:first-child{text-align:left}.w table{font-size:7pt}.note{font-size:7.5pt;text-align:justify;margin-top:2pt;line-height:1.15}
.r{font-size:8pt;line-height:1.15;padding-left:.27in;text-indent:-.27in;margin:0 0 1pt;break-inside:avoid}.r b{font-weight:400;display:inline-block;width:.27in;text-indent:0}
a{color:inherit;text-decoration:none}sub,sup{line-height:0}"""


def build_html():
    o = [f"<!doctype html><html lang='en'><meta charset='utf-8'><title>{html.escape(C.TITLE)}</title><style>{CSS}</style>",
         f"<div class='t'>{html.escape(C.TITLE)}</div><div class='a'>" + "<br>".join(f"<span{' style=font-size:11pt' if i == 0 else ''}>{html.escape(a)}</span>" for i, a in enumerate(C.AUTHORS)) + "</div><div class='c'>",
         f"<p class='abs'><i>Abstract</i>—{H(C.ABSTRACT)}</p><p class='abs'><i>Index Terms—{H(C.INDEX)}</i></p>"]
    n1 = n2 = nf = nt = ne = 0
    inlist = False
    for b in C.B:
        k = b[0]
        if inlist and k != "li":
            o.append("</ul>"); inlist = False
        if k == "h1":
            n2 = 0
            if b[1] in UNNUM:
                o.append(f"<h1>{b[1]}</h1>")
            else:
                n1 += 1; o.append(f"<h1>{ROMAN[n1]}. {b[1]}</h1>")
        elif k == "h2":
            n2 += 1; o.append(f"<h2>{chr(64 + n2)}. {b[1]}</h2>")
        elif k in ("p", "pc"):
            o.append(f"<p{' class=n' if k == 'pc' else ''}>{H(b[1])}</p>")
        elif k == "li":
            if not inlist:
                o.append("<ul>"); inlist = True
            o.append(f"<li>{H(b[1])}</li>")
        elif k == "eq":
            ne += 1; o.append(f"<div class='eq'><span>{H(b[1])}</span><span>({ne})</span></div>")
        elif k == "fig":
            nf += 1
            img = base64.b64encode((HERE / "figures" / b[1]).read_bytes()).decode()
            o.append(f"<figure class='{'w' if b[3] else ''}'><img alt='{html.escape(b[2])}' src='data:image/png;base64,{img}'><figcaption>Fig. {nf}. {H(b[2])}</figcaption></figure>")
        elif k == "table":
            _, cap, head, rows, wide, note = b
            nt += 1
            e = html.escape
            o.append(f"<div class='tb {'w' if wide else ''}'><div class='cap'>TABLE {ROMAN[nt]}<br>{e(cap)}</div><table><tr>" + "".join(f"<th>{e(x)}</th>" for x in head) + "</tr>" +
                     "".join("<tr>" + "".join(f"<td>{e(x)}</td>" for x in r) + "</tr>" for r in rows) + "</table>" + (f"<div class='note'>{e(note)}</div>" if note else "") + "</div>")
    if inlist:
        o.append("</ul>")
    o.append("<h1>References</h1>" + "".join(f"<p class='r'><b>[{i}]</b>{H(r)}</p>" for i, r in enumerate(C.REFS, 1)) + "</div></html>")
    return "".join(o)


def build_pdf(path):
    chrome = next(filter(None, map(shutil.which, ("google-chrome-stable", "google-chrome", "chromium"))), None)
    if not chrome:
        sys.exit("Chrome or Chromium is needed for the PDF (or export it from Word)")
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "paper.html"
        f.write_text(build_html(), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={t}/u", "--no-pdf-header-footer", f"--print-to-pdf={path}", f.as_uri()],
                       check=True, capture_output=True, timeout=90)


build_docx(HERE / "FactIndic_IEEE.docx")
build_pdf(HERE / "FactIndic_IEEE.pdf")
print("ok")
