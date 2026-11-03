"""Measure proposed character-role agreement; never equate it to correctness."""
from .contracts import ContractError


def role_mask(proposal):
    return [int(node['role']=='identity')for node in proposal['nodes']for _ in node['text']]


def compare_reviews(first,second):
    a=''.join(node['text']for node in first['nodes']);b=''.join(node['text']for node in second['nodes'])
    if a!=b:raise ContractError('Review proposals refer to different original source text')
    x=role_mask(first);y=role_mask(second)
    if not x:raise ContractError('Nonempty source required')
    n=len(x);observed=sum(a==b for a,b in zip(x,y))/n
    px=sum(x)/n;py=sum(y)/n;expected=px*py+(1-px)*(1-py)
    kappa=(observed-expected)/(1-expected)if expected!=1 else None
    tp=sum(a==b==1 for a,b in zip(x,y));fp=sum(a==0 and b==1 for a,b in zip(x,y));fn=sum(a==1 and b==0 for a,b in zip(x,y))
    return {'source_characters':n,'observed_character_agreement':observed,'cohen_kappa':kappa,
      'identity_character_f1':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,
      'same_full_payload':first==second,
      'scope':'Agreement, not annotation correctness or independent-review evidence; characters within a document are not independent trials.'}
