"""Oracle-based annotation safeguards and deliberately unsolved controls.

Two signing credentials are machine-generated test fixtures, not human reviewers.
The experiment cannot measure actual human annotation accuracy or agreement.
"""
import copy
import json
from pathlib import Path
import secrets
import sys
from fixtures import ROOT,fixture,NAMES,BODIES
from nameguard.annotations import endorse,packet,DualReviewAuthority,validate_payload
from nameguard.contracts import ContractError


def main():
    output=ROOT/'results/annotations.json'
    if output.exists():raise FileExistsError(output)
    keys={'review_A':secrets.token_bytes(32),'review_B':secrets.token_bytes(32)}
    authority=DualReviewAuthority(keys)
    def signed(p):return packet(p,[endorse(p,reviewer=n,key=k)for n,k in keys.items()])
    positives=[];omissions=[];disagreements=[]
    for body_id,body in enumerate(BODIES):
        for name in NAMES:
            p,_=fixture(body,name,0);doc=authority.verify(signed(p))
            original=''.join(node['text']for node in p['nodes'])
            if original!=''.join(node.text for node in doc.nodes):raise AssertionError('Source characters changed')
            oracle=[n['role']for n in p['nodes']]
            if oracle!=[n.role for n in doc.nodes]:raise AssertionError('Source role map changed')
            positives.append({'body_id':body_id,'name':name,'source_and_roles_exact_to_generator_oracle':True})
            for index,node in enumerate(p['nodes']):
                if node['role']!='identity':continue
                missed=copy.deepcopy(p);missed['nodes'][index]['role']='evidence'
                try:validate_payload(missed)
                except ContractError:omissions.append({'body_id':body_id,'name':name,'omitted_node':index,'blocked_before_inference':True})
                else:raise AssertionError('Known registered omission not rejected')
            alternative=copy.deepcopy(p);alternative['registry'].append('nonexistent-but-declared-name')
            mixed=packet(p,[endorse(p,reviewer='review_A',key=keys['review_A']),endorse(alternative,reviewer='review_B',key=keys['review_B'])])
            try:authority.verify(mixed)
            except ContractError:disagreements.append({'body_id':body_id,'name':name,'different_payload_blocked':True})
            else:raise AssertionError('Conflicting review passed')
    # Independent of corpus positives: preserve honest failure cases.
    from nameguard.annotations import proposal_from_spans
    unknown=proposal_from_spans('John, also known as Leaf, used Python.',[{'start':0,'end':4,'text':'John'}],['John'],resolved=True)
    unknown_doc=authority.verify(signed(unknown))
    shared_error=copy.deepcopy(unknown)
    # Both credentials can agree to retain an unregistered alias; signatures
    # provide no reason to regard this semantic error as impossible.
    unknown_survives='Leaf'in''.join(n.text for n in unknown_doc.nodes if n.role=='evidence')
    if not unknown_survives:raise AssertionError('Unknown-alias negative control was hidden')
    result={'positives':positives,'known_omission_controls':omissions,'review_disagreements':disagreements,
       'oracle_correct_cases':len(positives),'known_omissions_blocked':len(omissions),'disagreements_blocked':len(disagreements),
       'unknown_unregistered_alias_can_survive_even_two_signatures':unknown_survives,
       'human_annotation_accuracy_measured':False,'two_credentials_prove_two_independent_humans':False,
       'scope':'Mechanical safeguards evaluated with controlled exact roles; no universal semantic role correctness or independent-human annotation evidence.'}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['positives','known_omission_controls','review_disagreements']},indent=2))


if __name__=='__main__':main()
