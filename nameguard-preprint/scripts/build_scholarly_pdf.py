"""Render the tracked preprint Markdown with ReportLab; no model dependency."""
from html import escape
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,KeepTogether

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'output/pdf/nameguard-literature-preprint-draft.pdf';output.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PaperTitle',fontName='Times-Bold',fontSize=20,leading=24,spaceAfter=12,textColor=colors.HexColor('#142938')))
    styles.add(ParagraphStyle(name='PaperBody',fontName='Times-Roman',fontSize=10.5,leading=13.2,spaceAfter=6,splitLongWords=True))
    styles.add(ParagraphStyle(name='PaperH2',fontName='Times-Bold',fontSize=13,leading=16,spaceBefore=13,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#174b5a')))
    styles.add(ParagraphStyle(name='PaperH3',fontName='Times-Bold',fontSize=11.3,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperMeta',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=12,textColor=colors.HexColor('#52636b')))
    styles.add(ParagraphStyle(name='PaperCell',fontName='Helvetica',fontSize=8.5,leading=11))
    styles.add(ParagraphStyle(name='PaperCode',fontName='Courier',fontSize=8.5,leading=11,spaceBefore=4,spaceAfter=10,leftIndent=10))
    def markup(s):
        s=escape(s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
        s=re.sub(r'(https://[^\s<>]+)',r'<link href="\1" color="#174b5a">\1</link>',s)
        return s
    lines=(ROOT/'paper/manuscript.md').read_text().splitlines();story=[];paragraph=[];i=0
    def flush():
        if paragraph:
            story.append(Paragraph(markup(' '.join(paragraph)),styles['PaperBody']));paragraph.clear()
    while i<len(lines):
        line=lines[i]
        if not line.strip():flush();i+=1;continue
        if line.startswith('# '):
            flush();story.append(Paragraph(markup(line[2:]),styles['PaperTitle']));i+=1;continue
        if line.startswith('### '):
            flush();story.append(Paragraph(markup(line[4:]),styles['PaperH3']));i+=1;continue
        if line.startswith('## '):
            flush();story.append(Paragraph(markup(line[3:]),styles['PaperH2']));i+=1;continue
        if line.startswith('Adel Mansour |'):
            flush();story.append(Paragraph(markup(line),styles['PaperMeta']));i+=1;continue
        if re.match(r'^\d+\.\s',line):
            flush();paragraph.append(line);i+=1;continue
        if line.startswith('|'):
            flush();rows=[]
            while i<len(lines)and lines[i].startswith('|'):
                row=[s.strip()for s in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+',s)for s in row):rows.append([Paragraph(markup(s),styles['PaperCell'])for s in row])
                i+=1
            table=Table(rows,colWidths=([2.0*inch,4.0*inch] if len(rows[0])==2 else [6.0*inch/len(rows[0])]*len(rows[0])),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4eff1')),
              ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#749ba3')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe2'))]))
            story.extend([table,Spacer(1,10)]);continue
        if line.startswith('    '):
            flush();code=[]
            while i<len(lines)and(lines[i].startswith('    ')or not lines[i].strip()):
                code.append(lines[i][4:]if lines[i].startswith('    ')else'');i+=1
            story.append(Preformatted('\n'.join(code).rstrip(),styles['PaperCode']));continue
        paragraph.append(line);i+=1
    flush()
    def page(canvas,doc):
        canvas.saveState();width,height=doc.pagesize
        canvas.setStrokeColor(colors.HexColor('#a7bec5'));canvas.setLineWidth(.5)
        canvas.line(doc.leftMargin,height-41,width-doc.rightMargin,height-41)
        canvas.setFont('Helvetica',7.5);canvas.setFillColor(colors.HexColor('#52636b'))
        canvas.drawString(doc.leftMargin,height-32,'NAMEGUARD | LITERATURE-REVIEWED PREPRINT DRAFT')
        canvas.drawString(doc.leftMargin,31,'Conditional typed-identity guarantee; not universal raw-CV fairness.')
        canvas.drawRightString(width-doc.rightMargin,31,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=(612,792),leftMargin=72,rightMargin=72,topMargin=58,bottomMargin=55,
         title='Native-Token-Count Name-Channel Noninterference with Authenticated Role Review',author='Adel Mansour')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(output)


if __name__=='__main__':main()
