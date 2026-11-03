# Literature review: what name-channel isolation establishes

## Review method and boundaries

This is a structured narrative review, not an exhaustive systematic review or a
PRISMA study. Sources were checked on 3 November 2026 through publisher pages,
author manuscripts and official documentation. The 25 selected references cover
six questions: why audit names, which fairness property is intended, whether
behavior reveals internal information, what architectural exclusion guarantees,
how to preserve token efficiency, and how to evaluate a defensible artifact.
`search-log.json` records the search families, selection rules and exclusions.
`references.json` records source locations, version caveats and how extensively
each source was inspected. Abstract-only checks are identified rather than
presented as complete paper replications. The bibliography is in `references.bib`.

The review distinguishes a prior paper's findings from our deductions. A citation
motivates a design choice; it does not validate this implementation. We do not
infer the absence of other work from this targeted search. No numerical outcomes
from different models, tasks or populations are pooled into a meta-analysis.

## 1. Names as an experimental intervention in hiring

Bertrand and Mullainathan (2004) [6] establish a correspondence-audit precedent:
vary names associated with social groups while controlling resume content and
measure employer responses. This motivates matched-name interventions, not
assigning a person's actual race from a name. Our synthetic model-score checks
are not field evidence about employer callbacks.

Wilson and Caliskan (2024; corrected manuscript 2026) [7] audit resume retrieval
using text-embedding similarity. Their first-page erratum says gender-only
preferences were reversed by a coding error, while race-only and intersectional
results were unaffected. We therefore avoid the uncorrected abstract's gender
figures. Their retrieval setting differs from our autoregressive model readout;
its results motivate auditing but cannot be imported as our baseline bias rate.

Armstrong et al. (2024) [8] study GPT resume assessment and generation, showing why
evaluation should distinguish scoring existing evidence from producing new
candidate descriptions. We study a deterministic restricted scoring interface,
not generated recommendations or an end-to-end recruitment workflow. Together,
these audits justify controlled identity swaps while leaving deployment-wide
claims dependent on task, prompt, model and population.

## 2. Fairness is broader than an identity channel

Kusner et al. (2017) [3] formulate counterfactual fairness using a causal model.
Our equivalence relation instead fixes retained evidence Q and varies designated
identity text. We neither identify the causal effect of race nor hold constant
every consequence of discrimination. A job history or address can remain a
racial proxy even when applicant-name substitution cannot affect the score.

Dwork et al. (2012) [11] locate individual fairness in a task-relevant similarity
metric. Our exact equality of evidence is a deliberately narrower relation; it
does not solve how two different histories ought to be compared. Raghavan et al.
(2020) [9] examine hiring vendors' validation claims, making job relevance and
validation choices part of the research question rather than assumed outputs.
Selbst et al. (2019) [10] warn against abstractions that exclude the surrounding
social process. This means reviewer authority, applicant appeal, retained proxies
and selection policy belong in our limitations, not outside the system boundary.

The defensible conclusion is conditional name-channel noninterference, not
"truly unbiased hiring." Equal scores can be equally inaccurate. A harmful job
rubric stays harmful when its name input is disconnected. Real deployment needs
validity evidence, stakeholder review, accessible correction and subgroup audits
in addition to this narrow technical property.

## 3. Behavioral equality does not establish representation erasure

Hewitt and Liang (2019) [15] show why probe controls and selectivity matter: probe
success can partly reflect the probe's learning ability. Conversely, failure of
one fitted probe does not certify that no decoder can recover a concept. Gonen
and Goldberg (2019) [25] demonstrate residual gender structure after particular
word-embedding debiasing procedures. That finding concerns those embeddings;
we use it as a warning against generalizing one favorable bias metric.

Our simplest counterexample is H(q,n)=(q,n), with output S(H)=q. Swapping n cannot
change S, yet n is perfectly recoverable from H. Thus removing observable score
bias does not imply removing activation-space information. Even information
about names is not itself an unfair decision: knowing a name's linguistic origin
is compatible with prohibiting its influence on a hiring score. The distinction
is encoding versus causal use, under a stated intervention.

Amnesic probing (Elazar et al., 2021) [16] links representation interventions to
behavioral changes, rather than treating a probe alone as an explanation. Our
public-stage equality measurements similarly examine more than a final score,
but only after the explicit evidence boundary. Raw source embeddings, model
weights and any private diagnostic branch remain outside that erasure claim.

## 4. Erasure, adapters and structural routing are different interventions

INLP (Ravfogel et al., 2020) [13] repeatedly fits linear attribute predictors and
projects representations into their nullspaces. LEACE (Belrose et al., 2023;
manuscript v4, 2025) [14] derives a closed-form minimal-distortion linear eraser
under a precise guardedness definition. Its convex-loss linear-prediction
guarantee must not be restated as unrestricted independence or inability of
every zero-one-loss threshold classifier to recover a label.

For example, balanced Z with X=-1 or +1 when Z=0, and X=0 when Z=1 has equal
class means and zero covariance. Nevertheless, thresholding X>0 predicts Z=0
with 75% accuracy; a nonlinear decoder of |X| recovers Z exactly. This is our
finite mathematical witness, not a replication or refutation of LEACE's stated
theorem. It illustrates why the predictor family and loss must accompany the
word "erasure." Sample-estimated erasers also require held-out leakage tests.

LoRA (Hu et al., 2022) [17] freezes base weights and learns low-rank updates.
QLoRA (Dettmers et al., 2023) [18] combines frozen quantized weights with adapters
and specific memory-saving techniques. Our earlier MLX affine-four-bit adapter
experiments should be called quantized LoRA, not a faithful NF4/double-quantized
QLoRA replication. Neither adapter efficiency method by itself provides a
universal fairness guarantee. This final canonical-evidence release trains no
adapter and does not change base parameters.

Concept bottleneck models (Koh et al., 2020) [19] and ReFT (Wu et al., 2024) [20]
offer alternative controlled representations or interventions. They help frame
future learned evidence extractors, but are not evaluated baselines here.
The present tradeoff is explicit: deterministic isolation gives a conditional
dependence guarantee while shifting semantic correctness to reviewed roles.

## 5. The closest conceptual foundation is information-flow security

Sabelfeld and Myers (2003) [12] explain end-to-end information flow; Chong et al.
(2007) [4] distinguish confidentiality and integrity in an enforced system.
These are closer foundations for our theorem than statistical debiasing.
Let D~D' mean Q(D)=Q(D'). A score S=G(Q(D)) is constant on each equivalence class.
The proof is ordinary deterministic composition, not a new fairness theorem.
Authenticated reviews protect integrity of the chosen partition; they cannot
prove that the partition faithfully captures applicant identity.

This framing yields an essential baseline. An external canonicalizer computing
the identical Q followed by the same tokenizer, weights, positions and readout
has the same score as our internal canonicalizer. Embedding the raw source first
does not strengthen that output guarantee. It is a trace/design constraint and
adds work; names still exist in the raw private activations. Position, timing,
admission, error and audit channels need separate treatment. A correct public
score theorem is not a claim that the entire computer leaks no identity.

Internal placement is therefore an implementation characteristic, not evidence
of novelty or model unlearning. The artifact's contribution is an inspectable,
tested integration of reviewed source roles, native evidence tokenization and
frozen-model scoring, with explicit failure cases and reproducible records.

## 6. Tokenization efficiency and annotation reliability

Sennrich et al. (2016) [1] motivate subword segmentation for open-vocabulary NLP;
the official Tokenizers pipeline [2] separates transformations that can affect
boundaries. Our deduction is algebraic: a codec satisfying C(a+b)=C(a)||C(b) for
every scalar split is forced to encode scalars independently. It generally cannot
also preserve every merge of whole-string BPE. Joining reviewed evidence before
one tokenizer call avoids this incompatible requirement.

The exact result is n(Q)=|T(Q)| transformer positions, not globally minimum tokens
and not total compute parity with unrestricted input. Raw-source embedding and
review verification remain overhead. The measured 4.10x median paired speedup
compares our two service packages; it is not a speedup over a strong optimized
external-canonicalization baseline, which was not timed. It is also not an
isolated ablation of token count because review APIs differ.

Artstein and Poesio (2008) [5] distinguish agreement measures and interpretation.
Two credentials agreeing can share one error. Our synthetic oracle checks span
mechanics, registered omissions and conflicting payloads, not human semantic
accuracy. Independent adjudication must measure whole-document identity
omissions and false evidence removals, including unknown aliases, transliteration,
same-spelled skills and ambiguous third parties. Character agreement alone can
look excellent while missing a short identifying span.

## 7. Evaluation, reporting and research gaps

CheckList (Ribeiro et al., 2020) [21] motivates targeted behavioral tests alongside
aggregate accuracy. We separate name invariance, evidence sensitivity, annotation
failure controls, public-stage equality, native-evidence parity and efficiency.
The 108 controlled forwards repeat just three evidence bodies; their 3,360 array
comparisons are correlated engineering checks, not independent fairness trials.
Zero observed failures do not yield a population failure-rate estimate without
a sampling model. The universal conditional argument comes from architecture,
not extrapolation from those tests.

Dodge et al. (2019) [22] motivate reporting development choices and computation
budgets. We preserve the source freeze and distinguish inherited GPU measurements
from this later literature audit. Model cards (Mitchell et al., 2019) [23] and
datasheets (Gebru et al., 2021) [24] motivate explicit intended use, limitations
and data provenance. The final conformance corpus is generated; the supplied
public CV/job datasets belonged to earlier project exploration and do not supply
real-world validation for this extension.

Before an empirical fairness publication, the main missing evidence is an
independent real-document role study, a useful hiring-label benchmark and a
competitive external-canonicalization baseline. Before a systems publication,
the main missing evidence is a sharper threat model, implementation-level
verification or independent audit, and broader workloads/hardware. The current
artifact supports a transparent technical-report/preprint draft; novelty and
venue suitability remain reviewer judgments.

## Comparison of guarantees (our synthesis)

| Approach | What its result concerns |
|---|---|
| Name-swap hiring audits [6-8] | Measured outcomes on specific prompts, names and tasks; no universal erasure theorem |
| INLP / LEACE [13-14] | Specified linear representational criteria; not all decoders or every unseen distribution |
| Quantized LoRA / ReFT [17-18,20] | Efficient adaptation/intervention; fairness requires additional objectives and evaluation |
| Concept bottlenecks [19] | Predictions through supervised concepts; concept correctness remains an assumption |
| Information-flow enforcement [4,12] | Dependence under a policy and trusted boundary; the policy can be socially inadequate |
| This artifact | Equal scoring-path tensors for equal reviewed Q; no automatic semantic discovery or whole-model erasure |

## References

1. Rico Sennrich, Barry Haddow, Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. ACL. DOI: 10.18653/v1/P16-1162. https://aclanthology.org/P16-1162/ Accessed 2026-11-03.

2. Hugging Face. n.d.. The Tokenization Pipeline. Tokenizers documentation, version 0.20.3 (version pinned; publication date unspecified). https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline Accessed 2026-11-03.

3. Matt J. Kusner, Joshua R. Loftus, Chris Russell, Ricardo Silva. 2017. Counterfactual Fairness. NeurIPS. https://arxiv.org/abs/1703.06856 Accessed 2026-11-03.

4. Stephen Chong, K. Vikram, Andrew C. Myers. 2007. SIF: Enforcing Confidentiality and Integrity in Web Applications. USENIX Security. https://www.cs.cornell.edu/andru/papers/usenix07-html/paper.html Accessed 2026-11-03.

5. Ron Artstein, Massimo Poesio. 2008. Inter-Coder Agreement for Computational Linguistics. Computational Linguistics 34(4), 555-596. DOI: 10.1162/coli.07-034-R2. https://aclanthology.org/J08-4004/ Accessed 2026-11-03.

6. Marianne Bertrand, Sendhil Mullainathan. 2004. Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination. American Economic Review 94(4), 991-1013; NBER working paper 9873 (2003). DOI: 10.1257/0002828042002561. https://www.nber.org/papers/w9873 Accessed 2026-11-03.

7. Kyra Wilson, Aylin Caliskan. 2024. Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval. AIES 7(1); corrected author manuscript v3, 30 August 2026. DOI: 10.1609/aies.v7i1.31748. https://arxiv.org/pdf/2407.20371v3 Accessed 2026-11-03.

8. Lena Armstrong, Abbey Liu, Stephen MacNeil, Danaë Metaxa. 2024. The Silicon Ceiling: Auditing GPT's Race and Gender Biases in Hiring. ACM; author manuscript v3, 15 November 2024. DOI: 10.1145/3689904.3694699. https://arxiv.org/abs/2405.04412v3 Accessed 2026-11-03.

9. Manish Raghavan, Solon Barocas, Jon Kleinberg, Karen Levy. 2020. Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices. FAT*; author manuscript arXiv:1906.09208v3 (2019). https://arxiv.org/abs/1906.09208v3 Accessed 2026-11-03.

10. Andrew D. Selbst, danah boyd, Sorelle A. Friedler, Suresh Venkatasubramanian, Janet Vertesi. 2019. Fairness and Abstraction in Sociotechnical Systems. FAT*. DOI: 10.1145/3287560.3287598. https://friedler.net/papers/sts_fat2019.pdf Accessed 2026-11-03.

11. Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Richard Zemel. 2012. Fairness Through Awareness. ITCS; author manuscript arXiv:1104.3913 (2011). https://arxiv.org/abs/1104.3913 Accessed 2026-11-03.

12. Andrei Sabelfeld, Andrew C. Myers. 2003. Language-Based Information-Flow Security. IEEE Journal on Selected Areas in Communications 21(1). https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf Accessed 2026-11-03.

13. Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, Yoav Goldberg. 2020. Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. ACL. DOI: 10.18653/v1/2020.acl-main.647. https://aclanthology.org/2020.acl-main.647/ Accessed 2026-11-03.

14. Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman. 2023. LEACE: Perfect linear concept erasure in closed form. NeurIPS; author manuscript v4, 3 April 2025. https://arxiv.org/pdf/2306.03819v4 Accessed 2026-11-03.

15. John Hewitt, Percy Liang. 2019. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP. DOI: 10.18653/v1/D19-1275. https://aclanthology.org/D19-1275/ Accessed 2026-11-03.

16. Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. 2021. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. Transactions of the Association for Computational Linguistics. DOI: 10.1162/tacl_a_00359. https://aclanthology.org/2021.tacl-1.10/ Accessed 2026-11-03.

17. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR; author manuscript arXiv:2106.09685v2 (2021). https://arxiv.org/abs/2106.09685v2 Accessed 2026-11-03.

18. Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. 2023. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314 Accessed 2026-11-03.

19. Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang. 2020. Concept Bottleneck Models. ICML. https://arxiv.org/abs/2007.04612 Accessed 2026-11-03.

20. Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts. 2024. ReFT: Representation Finetuning for Language Models. Author manuscript arXiv:2404.03592. https://arxiv.org/abs/2404.03592 Accessed 2026-11-03.

21. Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL. https://arxiv.org/abs/2005.04118 Accessed 2026-11-03.

22. Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, Noah A. Smith. 2019. Show Your Work: Improved Reporting of Experimental Results. EMNLP-IJCNLP. https://arxiv.org/abs/1909.03004 Accessed 2026-11-03.

23. Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru. 2019. Model Cards for Model Reporting. FAT*; author manuscript arXiv:1810.03993. https://arxiv.org/abs/1810.03993 Accessed 2026-11-03.

24. Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, Kate Crawford. 2021. Datasheets for Datasets. Communications of the ACM; author manuscript arXiv:1803.09010. https://arxiv.org/abs/1803.09010 Accessed 2026-11-03.

25. Hila Gonen, Yoav Goldberg. 2019. Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them. NAACL-HLT. DOI: 10.18653/v1/N19-1061. https://aclanthology.org/N19-1061/ Accessed 2026-11-03.
