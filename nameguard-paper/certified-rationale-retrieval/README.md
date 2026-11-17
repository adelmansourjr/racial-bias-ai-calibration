# Exact sparse evidence selection on SciFact

In a replay on 209 expert-annotated SciFact claim/abstract pairs, sparse weighted alignment returned the same top two sentences in the same order as exhaustive scoring. With the same exact weighted-bag pruning, it reduced median ranking CPU by **35%** against dense alignment after shared corpus preprocessing. This study tests evidence selection inside a supplied abstract. The model-free scorer and its rankings are fixed across the exact implementations.

The experiment follows up on Ye, Santos-Rodríguez and Simpson's use of bounded intermediate checks for grounded factuality. It asks a complementary question about computation before factuality classification. [Grounded claim factuality checking, ACL 2026](https://aclanthology.org/2026.acl-long.1468/) · [SciFact, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.609/)

## First experiment

All arms read two selected sentences per case. Ranking combines normalized BM25 and IDF-weighted longest common subsequence. The official SciFact development data contributed 209 labelled claim/abstract pairs in 164 connected components. The study uses expert-annotated rationale sets in the candidate abstract and evaluates whether the chosen sentences cover a whole set. It conditions on receiving an evidence-containing abstract.

| Method | Exact ordered-list agreement | Dense alignment calls | Dense DP cells | At least one gold sentence | Complete gold rationale |
|---|---:|---:|---:|---:|---:|
| Exhaustive hybrid | 209/209 | 2,031 | 413,869 | 143/209 | 134/209 |
| Exact weighted-bag pruning | 209/209 | 470 | 105,465 | 143/209 | 134/209 |
| Exact ordered-pair pruning | 209/209 | 465 | 104,415 | 143/209 | 134/209 |
| BM25-first hybrid, matched alignment calls | 191/209 | 465 | 100,629 | 143/209 | 135/209 |
| Plain BM25 | 172/209 | 0 | 0 | 141/209 | 132/209 |

The ordered-pair bound saved 74.77% of dense cells over exhaustive scoring and 1.00% against the stronger weighted-bag pruner, missing the prespecified 10% target for the latter. Its extra bound calculation also takes more CPU. The fixed hybrid's complete-rationale rate was 0.96 percentage points above plain BM25 (paired bootstrap 95% interval −1.93 to +4.00) and 0.48 points below matched BM25-first (−2.93 to +2.14). Bootstrap clusters link cases that share a claim or a gold document. The matched policy knew each case's alignment-call budget. Its 18 changes to the ordered list illustrate the tradeoff between approximate scoring and exact preservation. [Protocol](PROTOCOL.md) · [Aggregate results](results.json)

For pruning, any valid alignment can use each token at most as often as it occurs in both texts. The bag bound sums those caps with the same term weights. The ordered-pair bound partitions the claim into adjacent blocks. Within each block, the bound can credit both tokens only if their order is feasible in the sentence, making the sum an upper bound. The algorithm stops when the best remaining bound, including the tie key, cannot displace the second-ranked candidate.

## Sparse alignment follow-up

The sparse kernel computes a maximum-weight increasing chain over matching claim/sentence token positions. Its Fenwick tree stores prefix maxima. Processing sentence matches in descending order for each claim token prevents using a single claim token twice. All 2,031 individual corpus alignment scores matched dense results. A separate match-graph implementation agreed with both kernels on 1,000 additional fixed-seed checks.

| Exact selection method | Dense ranking CPU | Sparse ranking CPU |
|---|---:|---:|
| Exhaustive | 38.72 ms | 12.76 ms |
| Weighted-bag pruning | 18.11 ms | **11.86 ms** |
| Ordered-pair pruning | 23.92 ms | 17.56 ms |

These are medians for the complete 209-case ranking pass over nine paired blocks on one ARM64 Mac with Python 3.12.8. Each block runs 12 passes per method in shuffled order. The ranking time includes query preparation, scoring, per-case bounds and exact alignment. Dense vs sparse under the same bag bound saved a median paired 35.1% and 8.5–37.0% across blocks. Shared archive loading and corpus indexing account for most of the cold-run cost. Adding that measured cost gives estimated medians of 314.09 ms (bag/dense) and 307.61 ms (bag/sparse). The sparse bag run processed 2,110 token-equality pairs with 10,258 total Fenwick query and update loop steps. These operation counts are distinct from dense DP cells. [Follow-up protocol](SPARSE_PROTOCOL.md) · [Sparse validation](sparse_results.json) · [CPU measurements](timing_results.json)

The performance claim is about the fixed lexical scorer and these two Python implementations, with exact preservation of its choices. The recall comparison is inconclusive. End-to-end verification accuracy, language-model computation, and the task of selecting the right abstract require separate evaluation.

## Reproduce

Use Python 3.12 and the official archive. The scripts require only the standard library, check the archive SHA-256 and store aggregate results. [SciFact dataset release](https://github.com/allenai/scifact)

```sh
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o data.tar.gz
python3 replay.py --archive data.tar.gz --verify results.json
python3 sparse_alignment.py --archive data.tar.gz --verify sparse_results.json
python3 sparse_alignment.py --archive data.tar.gz --timing
```
