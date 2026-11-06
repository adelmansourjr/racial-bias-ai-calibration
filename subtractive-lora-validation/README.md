# Subtractive LoRA validation on intact CV prompts

The learned correction reduced name-related score variation on held-out CVs.
It failed the exact-equality and qualification-retention gates. The role-bound
control gave exact equality on the same CVs and names using designated identity
spans.

| Held-out measure | Native | Prior subtractive adapter | New subtractive adapter | Role-bound control |
|---|---:|---:|---:|---:|
| Exactly equal serialized score contexts | 0/48 | 0/48 | 0/48 | **48/48** |
| Exactly equal full vocabulary contexts | 0/48 | 0/48 | 0/48 | **48/48** |
| Mean name-related logit range | 0.652 | 0.614 | **0.514** | 0 |
| Contexts above the 0.10-logit tolerance | 48/48 | 48/48 | 48/48 | 0/48 |
| Name-dependent decision flips | 3/48 | 2/48 | 2/48 | 0/48 |
| Jobs with variable top-three membership | 5/8 | 5/8 | 5/8 | 0/8 |
| Matched-category job wins | 14/24 | 14/24 | 14/24 | 16/24 |
| Synthetic qualification decisions correct | 59/96 | 53/96 | 53/96 | — |
| ARC-style multiple choice correct | 52/64 | 52/64 | 52/64 | — |

The learned adapter reduced the mean logit range by **21.2%**. A paired bootstrap
resampling CVs within occupation strata gives an exploratory change interval of
−0.156 to −0.121 logits. The largest remaining range was 1.297 logits. All
48 synthetic qualification pairs moved in the correct direction under all
three raw-input arms. The new adapter's binary rubric decisions still lost six
correct answers. The new adapter matched every native ARC-style choice in this
64-item sample. The qualification and ARC results are bounded diagnostics,
with no independent human hiring labels or formal noninferiority design.

## Experiment and implementation

The study used 24 CVs for fitting, 8 for development and 24 for the final
comparison. Each split had a distinct name pool. The final comparison crossed
two jobs per CV with six names for 288 intact named inputs. It used the
existing pinned Qwen model and the native vocabulary head. The selected
correction reads final-position states at blocks 8, 16, 24, and 28 and performs
one rank-one subtraction before scoring. A fit-only teacher supplied
evidence-only targets. The test and demo adapters receive named prompts.

The primary study and full role-bound check executed **1,144 model forwards**
across fitting, development, held-out tests and controls. Repeat-input checks
matched in all eight checked contexts. The role-bound check directly compared
480 counterfactual final hidden and full-vocabulary arrays, plus 96 arrays
against separate native evidence forwards. All these comparisons matched byte
for byte. Both experiments verified the pinned model parameter digest.

A separate analysis of earlier cached activations found 41.3% of held-out
name-contrast energy outside the fitted span, despite cancellation of the
fitting contrasts to numerical precision. This explains why fitting-set
cancellation alone supplies weak evidence for new names.

The role-bound result applies to the reviewed identity span with fixed evidence.
The learned raw-input adapter and the role-bound scorer have different input
contracts. See [the protocol and mathematical scope](METHOD.md).

## Inspect and reproduce

- [Verified primary summary](results/verified-summary.json)
- [Cluster and shortlist analysis](results/secondary-audit.json)
- [Exact role-bound comparison](results/exact-control.json)
- [Fit-span diagnostic](results/span-diagnostic.json)
- [Frozen fitting plan](audit/plan.json) and [development selection](audit/selection.json)

From the repository root, verify the public artifacts with:

```sh
python3 -m unittest discover -s subtractive-lora-validation/tests -v
python3 subtractive-lora-validation/verify_public.py
```

With local records, run `audit_results.py` to reconstruct the equality counts,
qualification results, acceptance gates and shortlist extrema. A full rerun
requires the original pinned checkpoint, local CV and job snapshots, historical
cohort files and prior adapter. The experiment refuses to overwrite its frozen
plan. Use an isolated workspace copy for another run. Dependencies are listed
in the [tested MLX requirements](../nameguard-validation/requirements-mlx.txt).

With the local saved coefficients, replay an intact named A/B prompt:

```sh
python3 subtractive-lora-validation/demo.py \
  --checkpoint /path/to/pinned/qwen \
  --prompt-file /path/to/named-prompt.txt
```

The published artifacts contain source and aggregates. Learned coefficients,
private input text, arrays and per-CV records are held locally. The independent
audit corrected the test drift field in the verified summary. The native-relative
mean drift is 0.175 logits for the prior adapter and 0.255 logits for the new
adapter. The frozen run's selection and acceptance outcomes agree with the audit.
