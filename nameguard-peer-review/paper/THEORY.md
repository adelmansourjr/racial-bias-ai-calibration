# Admission-aware noninterference and finite-panel contract compilation

## Scope and notation

A request x contains original text, a reviewed applicant-identity interval, and
retained evidence. T is the pinned native tokenizer. A valid candidate boundary
expands the identity interval by at most b=16 adjacent whitespace characters per
side, changes no source character, and satisfies native factorization:

    T(x) = T(evidence prefix) ++ T(identity interval) ++ T(evidence suffix).

The final slot must be retained evidence and the original sequence length must
meet the runtime limit. The set C(x) contains all such structural witnesses.
Each witness c supplies exact retained token sequence e(c) and nonnegative
formatting cost k(c). The policy permits only explicitly reviewed formatting
reclassification. It does not define the social meaning of a name automatically.

For a finite declared panel x_1,...,x_n, let A_i={e(c): c in C(x_i)}. A single
fixed contract e structurally supports x_i precisely when e is in A_i. At inference, the
contract is fixed by the operator; the incoming name cannot choose a different
contract. Every chosen witness still requires semantic approval.

Structural support is an upper bound on actual admission if semantic review can
reject candidates. Let C_approved(x) be the subset with correct source-bound
semantic approval. Actual-admission optimality requires optimizing those approved
sets. This implementation optimizes structural witnesses and returns unresolved
proposals. In the synthetic experiment the generator authorizes the formatting
policy. No equivalent optimum after human review of real CVs is claimed. The
selector assumes valid trusted Candidate objects produced by enumeration: finite
nonnegative costs, native token IDs and correctly bound source offsets. It is
an internal research helper, not a validator for untrusted candidate objects.

## Proposition 1: complete decision-and-score outcome invariance

Let a(x) be the deterministic admission indicator and f(x) the public score when
admitted. Define the decision-and-score outcome O(x)=REJECT if a(x)=0, and
O(x)=(ACCEPT,f(x)) otherwise. Then O is invariant over a substitution family X
if and only if:

1. a is constant over X; and
2. whenever a=1, f is constant over X.

This observer excludes timings, token counts, distinct rejection reasons and
other metadata. A deployed interface exposing these values has a larger observer
and needs additional equalities. The theorem concerns the defined tagged score
outcome, not every observation of the process or API.

Proof. Equal tagged outcomes imply equal tags and, for accepted tags, equal
scores. Conversely these two equalities give identical tagged outcomes.
All-rejection is invariant but has zero service coverage and is never reported
as a successful scoring solution. This proposition is elementary: its role is
to expose a missing endpoint in accepted-only fairness reports, not to claim a
new general theory of selective classification.

For distributions of names P and Q, total variation between complete-outcome
distributions is at least |Pr_P(ACCEPT)-Pr_Q(ACCEPT)|. Proof: the ACCEPT event is
one event in the supremum defining total variation. Consequently, perfect
conditional score parity does not eliminate an admission disparity. We measure
name-string conventions, not racial-group distributions; no demographic claim
is inferred from script labels.

## Proposition 2: maximal finite-panel support

Define support s(e)=sum_i 1[e in A_i]. Let E be the union of A_i. If E is empty,
no request can be admitted under this single-contract policy. Otherwise choose

    e* = argmin_e in E (-s(e), K(e), lex(e)),
    K(e) = sum_{i:e in A_i} min_{c:e(c)=e} k(c).

Then e* maximizes structural support among all single-contract choices.
Among maximizers it minimizes total witness-formatting cost. The final token-ID
ordering makes the choice deterministic. Repeated candidates from one panel
member count only once.

Proof. A contract outside E admits nobody. For every contract in E, the support
count equals precisely the number it structurally supports. Exhaustive comparison of these
counts therefore finds the maximum. Minimizing each supported member's
formatting cost independently minimizes their sum; lexicographic selection
resolves remaining ties. This is a finite maximum-support construction using
standard enumeration, rather than a new combinatorial optimization theorem.

With at most (b+1)^2 witnesses per panel member, candidate enumeration considers
at most n(b+1)^2 boundary pairs. Tokenization cost is additional, dependent on
source length and tokenizer implementation. A dictionary keyed by exact token
tuples accumulates support; Python hash collisions are resolved by tuple
comparison. No cryptographic collision assumption is required for admission.

## Corollary 2.1: exact coverage ceiling under the declared policy

A single contract structurally supports every member if and only if
intersection_i A_i is nonempty. Otherwise at least n-max_e s(e) members must
lack a structural witness under this policy. Define maximum support as zero if
E is empty. No learned adapter can change that feasibility fact
while retaining the same input/token-boundary policy and single-contract
requirement. A different architecture or review policy can change A_i, so the
ceiling is not a universal impossibility result for all fairness methods.

A finite panel certificate is not a proof about every future name. Fitting e*
on a development name panel supplies a fixed contract; future names undergo an
exact witness check. Generalization of acceptance is empirical, while equality
of admitted scores follows from the structural theorem below. Test-panel oracle
optimization is reported only as a descriptive upper bound and never used to
set the primary held-out contract.

## Proposition 3: inherited internal-state isolation

Let G_x gather reviewed evidence rows and S_x scatter them into original slots,
with zero identity rows. Let native block B_l receive the gathered evidence at
fixed evidence-relative positions with the same causal mask. For equal contract
e and fixed frozen parameters:

    A_0=Embedding(e), A_(l+1)=B_l(A_l), H_l=S_x(A_l), G_x S_x=I.

Every admitted variant has equal A_l by induction, hence equal final vocabulary
arrays and scores under the deterministic runtime. Identity rows are zero after
the boundary. The original named embeddings before that boundary and the full
stored layout can differ. This result reuses the earlier implementation's
noninterference construction and classical information-flow reasoning; it is
not a new claim of model-wide name amnesia.

Complete-outcome invariance over the panel follows only when the same contract
admits all members. A partial-coverage panel has a scoring noninterference
certificate for admitted members and an explicitly failed complete-outcome gate.

## Proposition 4: a decisive equivalent baseline

An external method that passes the exact same evidence token sequence e to the
same native blocks, positions, mask and head has the same scoring states as the
internal method. Internal placement alone cannot improve mathematical fairness
relative to that equivalent baseline. This is tested directly at all aligned
stages. Internal routing additionally embeds original tokens and maintains full
physical storage, which may carry extra cost.

Deleting name characters and retokenizing the remaining text is a different
baseline. T(prefix ++ suffix) need not equal T(prefix) ++ T(suffix). If token
sequences differ, identical outputs no longer follow from Proposition 3. This
comparison separates intervention placement from a changed evidence encoding.

## Statistical interpretation

For n i.i.d. Bernoulli trials with a common failure probability and zero failures, the exact one-sided
1-alpha upper failure-rate bound is 1-alpha^(1/n). It follows by solving
(1-p)^n=alpha. Repeated names, related prompt styles and deliberately chosen
layouts are not independent population samples; plugging every repeated row
into this formula would exaggerate evidence. We supply the formula and tests
but do not present a population guarantee from the synthetic panels.

Task comparisons aggregate over eight credential task families. Within each
family, states, names and wording styles are repeated measures. The percentile
bootstrap resamples these eight family-level paired differences; its interval
is descriptive and weak at this small number of constructed families. Exact
array equalities support implementation conformance rather than statistical
proof of semantic fairness or independent replication.

## Proposed contribution and boundary of novelty

The new project contribution is an explicit admission-aware contract compiler
and reproducible audit joining native BPE boundary feasibility with internal
score isolation. It adds finite support certificates, disjoint-name acceptance
tests, exact same-ID baselines and complete-outcome reporting. Maximum support,
noninterference and abstention fairness have substantial prior work. Priority
for this specific application is not proven by our targeted literature search.
The work should be reviewed as a focused systems/audit study, not a new universal
unlearning theorem.

## Proposition 5: exact native fidelity has an unavoidable lower bound

For a fixed CV/job context, finite real native name scores v_1,...,v_n, and
any finite invariant replacement c, let mean(v) be their arithmetic mean and
variance(v)=n^(-1) sum_i (v_i-mean(v))^2. The positive decision convention is
v >= 0, including equality at the threshold. Then

    mean_i (v_i-c)^2 = variance(v) + (mean(v)-c)^2.

Proof. Expand v_i-c=(v_i-mean(v))+(mean(v)-c); the cross term averages to zero.
Thus the minimum mean-squared deviation from native scores is variance(v),
achieved by their mean. Also

    min_c max_i |v_i-c| = (max(v)-min(v))/2,

achieved by the midpoint of the extrema. The lower bound follows because both
extreme scores cannot be within less than half their separation of c.

At a zero decision threshold, any deterministic invariant binary rule must
change at least min(n_positive,n_negative) native decisions in that context;
choose the majority native label to attain the bound. These are standard
least-squares, minimax-center and majority-vote identities, applied here to
quantify fidelity to a name-sensitive model. We do not claim them as new general
mathematical discoveries.

This establishes why retaining every original name-dependent score and making
all names equal are incompatible. It does NOT establish a necessary loss in
accuracy against a name-independent qualification label. A constant equal to the
correct qualification label could have perfect task accuracy. The measured task
accuracy loss must therefore be reported as an empirical limitation of this
readout, rather than excused as an unavoidable fairness-accuracy tradeoff.
