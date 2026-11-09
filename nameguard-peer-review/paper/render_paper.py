"""Render the review manuscript as a readable, numbered technical-report PDF."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Preformatted,Table,TableStyle,KeepTogether,PageBreak
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'output/pdf/nameguard-admission-aware-preprint.pdf'


def inline(text):
    text=escape(text).replace('–','-').replace('—','-').replace('−','-')
    return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)


def footer(canvas,doc):
    canvas.saveState()
    w,h=doc.pagesize
    canvas.setStrokeColor(colors.HexColor('#CAD4DE'));canvas.line(49,h-40,w-49,h-40)
    canvas.setFont('Times-Italic',8.5);canvas.setFillColor(colors.HexColor('#43566A'))
    canvas.drawString(49,h-31,'Admission-aware name noninterference | Technical-report draft')
    canvas.setFont('Times-Roman',8.5);canvas.drawString(49,29,'9 November 2026 | Automated reviews disclosed; human validation pending')
    canvas.drawRightString(w-49,29,str(doc.page));canvas.restoreState()


def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PaperTitle',fontName='Times-Bold',fontSize=22,leading=25,spaceAfter=13,textColor=colors.HexColor('#123B52')))
    styles.add(ParagraphStyle(name='PaperBody',fontName='Times-Roman',fontSize=10.6,leading=14.1,spaceAfter=8))
    styles.add(ParagraphStyle(name='PaperH1',fontName='Times-Bold',fontSize=14,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#123B52')))
    styles.add(ParagraphStyle(name='PaperH2',fontName='Times-Bold',fontSize=11.5,leading=14.5,spaceBefore=9,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperRef',fontName='Times-Roman',fontSize=9.1,leading=12.2,spaceAfter=7,splitLongWords=True))
    styles.add(ParagraphStyle(name='PaperTable',fontName='Times-Roman',fontSize=9.8,leading=12.5))
    styles.add(ParagraphStyle(name='PaperCode',fontName='Courier',fontSize=8.5,leading=11.5,leftIndent=13,spaceBefore=3,spaceAfter=9))
    doc=SimpleDocTemplate(str(TARGET),pagesize=(595.28,841.89),leftMargin=51,rightMargin=51,topMargin=54,bottomMargin=49,
                          title='Beyond Equal Scores: Admission-Aware Name Noninterference in Frozen Transformer Scoring',author='Adel Mansour (details to confirm)')
    lines=(ROOT/'paper/manuscript.md').read_text().splitlines();story=[];i=0;refs=False
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['PaperTitle']));i+=1;continue
        if line.startswith('## '):
            refs=line=='## References'
            if refs:story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]),styles['PaperH1']));i+=1;continue
        if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['PaperH2']));i+=1;continue
        if line.startswith('    '):
            code=[]
            while i<len(lines) and (lines[i].startswith('    ') or not lines[i].strip()):
                code.append(lines[i][4:] if lines[i].startswith('    ') else '');i+=1
            story.append(Preformatted('\n'.join(code).rstrip(),styles['PaperCode']));continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',v) for v in row):
                    rows.append([Paragraph(inline(v),styles['PaperTable']) for v in row])
                i+=1
            table=Table(rows,colWidths=[238,117,137],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF3')),('VALIGN',(0,0),(-1,-1),'TOP'),
               ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8092A2')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D4DEE6')),
               ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,9)]);continue
        para=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','    ','|')):
            para.append(lines[i].strip());i+=1
        text=' '.join(para)
        item=Paragraph(inline(text),styles['PaperRef' if refs else 'PaperBody'])
        story.append(KeepTogether([item]) if text.startswith('Two fresh-context AI agents') else item)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(TARGET)


if __name__=='__main__':main()
