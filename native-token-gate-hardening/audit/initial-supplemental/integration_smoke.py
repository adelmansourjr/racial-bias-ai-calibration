"""Exercise fixed task profiles and evidence contracts on the actual GPU service."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets

from gate import SlotScorer, ContractError
from service import CredentialScoringService, load_credential_calibration
from evidence_contract import EvidenceContract, BoundEvidenceScorer
from nameguard.annotations import DualReviewAuthority, endorse, packet, proposal_from_spans
from nameguard.mlx_backend import load_pinned

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    args = parser.parse_args()
    target = ROOT/'results/service-smoke.json'
    if target.exists():
        raise FileExistsError(target)
    model,tok,verify_files = load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    keys = {key:secrets.token_bytes(32) for key in ('a','b')}
    authority = DualReviewAuthority(keys)
    raw = SlotScorer(model,tok,authority)
    artifact = ROOT/'audit/calibration.json'
    fitted = load_credential_calibration(artifact,hashlib.sha256(artifact.read_bytes()).hexdigest())

    def wire(name, present, skill='CPA'):
        body = ('Applicant name:\n'+name+'\nRole: accountant. The sole requirement is an active '+
                skill+' credential. The applicant '+('has' if present else 'does not have')+
                ' an active '+skill+' credential. Answer A if the requirement is met. '
                'Otherwise answer B. Answer only A or B.')
        source = tok.apply_chat_template([{'role':'user','content':body}],
                                         tokenize=False,add_generation_prompt=True)
        start = source.index('Applicant name:\n')+len('Applicant name:\n')
        p = proposal_from_spans(source,[{'start':start,'end':start+len(name),'text':name}],[name],resolved=True)
        return packet(p,[endorse(p,reviewer=k,key=v) for k,v in keys.items()])

    matches = decisions = 0
    for present in (False,True):
        baseline = wire('Ai',present)
        contract = EvidenceContract.from_reviewed(baseline,authority,tok)
        protected = CredentialScoringService(BoundEvidenceScorer(raw,contract),fitted)
        scores = [protected.score(wire(name,present)) for name in ('Ai','Inês de Albuquerque-Silva')]
        if scores[0] != scores[1]:
            raise AssertionError('Integrated service name equality failed')
        if (scores[0].margin>=0) != present:
            raise AssertionError('Integrated service qualification decision failed')
        matches += 1
        decisions += len(scores)
        rejected = False
        try:
            protected.score(wire('Ai',present,skill='HVAC'))
        except ContractError:
            rejected = True
        if not rejected:
            raise AssertionError('Changed evidence was admitted under a stale CV/job contract')
    digest = raw.verify_parameters()
    verify_files()
    raw.close()
    result = {'exact_short_and_long_name_pairs':matches,'correct_qualification_decisions':decisions,
              'changed_evidence_contracts_rejected':2,'raw_backend_forwards':4,
              'parameter_digest':digest,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
