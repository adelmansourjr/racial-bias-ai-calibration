# NameGuard article holdout and operating limits

Frozen before executing this retrospective follow-up. The 320 reserve cases, labels and aggregate results were already inspected in the published NameGuard study. No new model calls or labels are collected. Group by SQuAD source article so no question or passage from a held-out article can inform threshold fitting within its fold.

## Main test

Reuse the candidate-aware score and candidate-blind exact extraction flag from the archived reserve predictions. Compare score-only thresholding against thresholds that admit only exact extractions, with the threshold learned separately on the other nine articles. An eligible threshold must retain at least 20% of training cases and show at most 10% false support among confirmations. Consider every distinct eligible training score; choose the eligible threshold with highest coverage, breaking ties by fewer errors and then higher threshold. If none qualifies, refer the entire held-out article to review. Primary outcome is whether the pooled held-out decisions clear both targets. Zero coverage has undefined false-support risk and fails. The training screen uses empirical proportions and provides no statistical guarantee. Report every article and the number of folds without an eligible threshold.

## Workload-matched diagnostic

For every article compare two orderings at the same capacity of floor(20% of its cases). Use the candidate-aware score alone or prioritise exact extractions then use the same score. Break score ties by public case ID. Both orderings are label-free on that article. Also report the original exact-extraction rule at its naturally resulting workload. The capacity-forced policies can accept unsupported cases and are comparative diagnostics, not recommended operating rules. Report paired article intervals using 10,000 resamples, seed 3750, as conditional summaries for the fixed cohort, alongside all article contrasts. They do not establish performance for new articles.

An oracle subset bound for an extraction-only filter counts the maximum possible accepted items at 10% false support given the total true positives among the extracted cases. It establishes a limit on this archived set even if the filtering rule knew the labels. It says nothing about a different generator or a rule admitting additional cases. Publish aggregate outcomes only. Underlying public questions and labels can be reconstructed using the original pinned SQuAD source and replay instructions.
