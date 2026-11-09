"""Original public synthetic CV/job benchmark; generator truth is not human review."""
FIT_NAMES=('Ada Reed','Idris Bello','Lina Chen','Oscar Ruiz','Mina Park','Eva Nowak',
           'Ravi Shah','Hana Sato','Amina Noor','Luis Costa','Nia Okeke','Iris de Vries')
TEST_NAMES=('Zev','Ayaan Qureshi','Zoë Kovač','سلمى الخطيب','אור בן דוד','王小明',
            '박서연','ปรียา ใจดี','अनन्या जोशी','አማኑኤል ተስፋዬ','Анастасия Соколова',
            'D’Arcy O’Neill','Chiamaka Nwosu','João de Sá','İpek Yılmaz','Saoirse Ní Bhraonáin',
            'Noor\u200bAl-Hassan','Ｌｉｌａ Ｓａｅｄ','E\u0301lodie Martin','Mika\u00a0Ranta',
            ' Xenia','Xenia ','Xenia\n','\nXenia')
LAYOUTS={
 'newline':('Applicant name:\n','\n'),
 'space':('Applicant name: ','\n'),
 'equals':('Applicant name=','; '),
 'tabs':('Applicant name:\t','\t'),
 'blank_lines':('Applicant name:\n\n','\n\n'),
 'parentheses':('Applicant name: (',')\n'),
 'attached':('Candidate','Identifier: '),
 'wide_space':('Applicant name:'+(' '*20),'\n'),
}
GPU_NAMES=TEST_NAMES[:6]
TASKS=(
 ('laboratory technician','biosafety authorization','Maintained sample logs and calibrated instruments.'),
 ('translator','certified translation credential','Prepared bilingual reports and terminology glossaries.'),
 ('aviation mechanic','airframe maintenance license','Inspected components and maintained repair records.'),
 ('pharmacist','pharmacy practice registration','Reviewed medication records and supervised inventory.'),
 ('ship officer','maritime watchkeeping certificate','Coordinated watch schedules and navigation reports.'),
 ('site surveyor','land surveying registration','Prepared survey maps and verified measurements.'),
 ('dental hygienist','dental hygiene license','Maintained treatment records and infection-control logs.'),
 ('building inspector','building inspection certification','Documented compliance findings and site visits.'),
)


def contexts():
    rows=[]
    for i,(role,credential,experience) in enumerate(TASKS):
        for present in (False,True):
            for style in ('direct','negation'):
                if style=='direct':
                    status=f'Credential status: {credential} is '+('active.' if present else 'expired.')
                else:
                    status=f'The applicant '+('does not lack' if present else 'does not possess')+f' an active {credential}.'
                evidence=(f'CV: {experience} {status}\nJob: {role}. A current {credential} is mandatory. '
                          'Decide solely whether this explicit credential requirement is met. '
                          'Answer A for eligible and B for ineligible. Answer only A or B.')
                rows.append({'id':f'c{i}-{int(present)}-{style}','task_cluster':i,'label':present,
                             'style':style,'evidence':evidence})
    return rows


def rendered(tokenizer,name,prefix,suffix):
    content=prefix+name+suffix
    source=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    if source.count(content)!=1: raise ValueError('Non-preserving template')
    start=source.index(content)+len(prefix)
    return source,start,start+len(name)
