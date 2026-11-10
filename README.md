# From Paths to Answers: Budgeted Knowledge-Graph Question Answering

Reproduction code for **From Paths to Answers: Separating Selection and Reading in Budgeted Knowledge-Graph Question Answering**.

The repository contains the current source code, executable setup and analysis helpers, synthetic tests, and this README. Benchmark data, model weights, predictions, manuscript files, figures, and private verification records are not distributed.

## Quick start

```sh
git clone https://github.com/xywang815/budgeted-kgqa.git
cd budgeted-kgqa
python3 scripts/setup_workspace.py
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python scripts/install_environment.py --mode analysis --install
.venv-analysis/bin/python scripts/check_environment.py --mode analysis
.venv-analysis/bin/python -m unittest discover -s tests -v
```

Workspace setup downloads nothing and refuses to overwrite an existing workspace. Obtain and link the source data, prepare the GPU environment and models, run inference, and then run CPU analysis, as described below. The experiment plan prints commands until `--run` is supplied.

## Data and model sources

Official source pages rechecked on 10 November 2026. Original benchmark distributions, full graph neighborhoods, and model weights are not stored in this repository.

| Resource | Official source | Role in this study |
|---|---|---|
| MetaQA | https://github.com/yuyuz/MetaQA | Vanilla 1-, 2-, and 3-hop questions and movie knowledge graph; use the provider's dataset download link |
| WebQuestionsSP (WebQSP) | https://www.microsoft.com/en-us/download/details.aspx?id=52763 | Official questions, semantic parses, and scoring program |
| UniK-QA resources | https://github.com/facebookresearch/UniK-QA | Freebase two-hop neighborhoods for the WebQSP question IDs |

UniK-QA's official README links to `https://dl.fbaipublicfiles.com/UniK-QA/data.tar.xz`. This large resource is needed for fresh WebQSP inference, not the CPU audit. Follow the provider's terms and retain its `stagg.neighborhoods/*.nxhd` layout.

The official WebQSP release is available from the Microsoft page above. Its archive is `WebQSP.zip`; the original scoring program is included in that release. This project's best-parse sensitivity code applies the same best-F1-over-parses rule to its selected question subset.

### Download and connect the required files

Perform large downloads on the inference server. The MetaQA repository links to the provider's Google Drive folder; cloning the repository alone does not download the dataset. Obtain `kb.txt` and `1-hop/vanilla`, `2-hop/vanilla`, `3-hop/vanilla` with `qa_dev.txt` and `qa_test.txt`. Audio, NTM, and MetaQA training files are not needed by this study. The post-review semantic controls also use the official WebQSP training file and training neighborhoods.

The WebQSP release is `WebQSP.zip` from the Microsoft Download button. Extract it and locate `WebQSP/data/WebQSP.test.json` and `WebQSP/data/WebQSP.train.json`. UniK-QA's large archive supplies `data/freebase_2hop/stagg.neighborhoods/`; preserve its `WebQTest-*.nxhd` and `WebQTrn-*.nxhd` files. The original neighborhood extraction is required; another Freebase graph is not interchangeable.

After extraction, use your actual absolute paths in place of the examples:

```sh
python3 scripts/link_resources.py --workspace work/reproduction --metaqa /absolute/path/to/MetaQA --webqsp-test /absolute/path/to/WebQSP/data/WebQSP.test.json --webqsp-train /absolute/path/to/WebQSP/data/WebQSP.train.json --neighborhoods /absolute/path/to/data/freebase_2hop/stagg.neighborhoods
python3 scripts/verify_resources.py --workspace work/reproduction
```

The helper links the provider's root `kb.txt` into the preprocessor's expected `data/raw/metaqa/kb/kb.txt`, links the six vanilla question files, and makes both required WebQSP test-file paths. It does not modify provider data or copy large graphs. `--scope metaqa` and `--scope webqsp` allow separate checks. Seven MetaQA source fingerprints and the official WebQSP test fingerprint are embedded in the verifier; these hashes are metadata, not distributed data. A changed newline format or different release fails the fingerprint check.

MetaQA preprocessing removes exact duplicate triples while preserving order and applies the audited adjacent-fragment repair to two one-hop development and three one-hop test records. It retains raw and corrected answers and reports every correction. This is why the original bytes and preprocessing script matter. The WebQSP resource audit checks usable official parses and topic coverage; the common-pool runner verifies its 1,577-question eligibility set and 500/1,077 split. No unverified full-archive hash is claimed for the large UniK-QA download.

### Model revisions

| Model repository | Recorded revision |
|---|---|
| https://huggingface.co/Qwen/Qwen3-8B-AWQ | `cb7d6a337aadb4d2082ed0dcef1032e4f8645194` |
| https://huggingface.co/microsoft/Phi-4-mini-instruct | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| https://huggingface.co/intfloat/e5-small-v2 | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |

Use the tokenizer stored with each recorded model snapshot. Qwen thinking is disabled; its AWQ weights use float16 computation. Phi uses bfloat16. Frozen E5 uses float32, query/passage prefixes, masked mean pooling, L2 normalization, and a 512-token encoder limit. These are different from the evidence-token caps applied after selection.

### Local inputs and outputs

Download source data under a local workspace and retain the provider's split and graph layout. Model snapshots must use the revisions above. Sampling IDs, predictions, token counts, and statistics are produced locally by the experiment scripts. They are not included in this repository.

### Source citations

- Zhang, Y.; Dai, H.; Kozareva, Z.; Smola, A.; Song, L. Variational Reasoning for Question Answering with Knowledge Graph. AAAI, 2018. https://doi.org/10.1609/aaai.v32i1.12057
- Yih, W.-t.; Richardson, M.; Meek, C.; Chang, M.-W.; Suh, J. The Value of Semantic Parse Labeling for Knowledge Base Question Answering. ACL, 2016. https://doi.org/10.18653/v1/P16-2033
- Oguz, B. et al. UniK-QA: Unified Representations of Structured and Unstructured Knowledge for Open-Domain Question Answering. Findings of NAACL, 2022. https://doi.org/10.18653/v1/2022.findings-naacl.115

Benchmark-derived names, questions, and annotations retain their original providers' terms. A repository link does not replace source attribution or alter dataset/model licenses.

## Two execution environments

Run all commands below from the repository root. Use separate virtual environments: the historical inference runtime used NumPy 2.1.3; the publication analysis is pinned to NumPy 2.3.5. Installing analysis requirements into the inference environment would change that runtime.

### CPU analysis (Linux or macOS, Python 3.11/3.12)

```sh
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python scripts/install_environment.py --mode analysis --install
.venv-analysis/bin/python scripts/check_environment.py --mode analysis
.venv-analysis/bin/python -m unittest discover -s tests -v
```

This environment runs statistics, synthetic workflow tests, and figures. It does not run Qwen, Phi, or E5 inference.

### Historical inference (Linux x86_64, Python 3.12, NVIDIA GPU)

The recorded machine used Ubuntu 22.04, Python 3.12.3, an RTX 4090 with 24 GiB, NVIDIA driver 560.35.03, and the CUDA 12.4 PyTorch runtime. The Qwen and E5 models run together in the shared-pool experiment. Phi requires bfloat16 support.

```sh
python3.12 -m venv .venv-inference
.venv-inference/bin/python -m pip install --upgrade pip
.venv-inference/bin/python scripts/install_environment.py --mode inference --install
.venv-inference/bin/python -m pip check
.venv-inference/bin/python scripts/check_environment.py --mode inference
```

The PyTorch command follows the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/). Core inference dependencies are pinned to the original recorded versions. AutoAWQ 0.2.9 and kernels 0.0.9 are retained for the original AWQ implementation. `check_environment.py` imports both the AWQ implementation and compiled extension and checks CUDA/bfloat16 availability; `shared-smoke` then exercises the actual tokenizer, E5, Qwen, and budget checks.

The installation recipe follows the recorded runtime and package dependency metadata. Follow-up model controls were executed in the retained Linux/CUDA environment after model, tokenizer, and helper hashes were verified. The setup recipe has not been tested through a fresh CUDA installation on a second machine. If a binary import fails, check the pinned Python, Torch, CUDA wheel, and kernel versions before changing them. Preserve logs and package versions for any alternative environment.

### Pinned model downloads (on the inference server)

Create the workspace first, then preview or execute the downloads:

```sh
.venv-inference/bin/python scripts/download_models.py --workspace work/reproduction
.venv-inference/bin/python scripts/download_models.py --workspace work/reproduction --download
```

Use `--model qwen`, `--model phi`, or `--model e5` to download one snapshot. The script pins each repository revision, downloads safetensors and tokenizer/configuration files, and avoids duplicate ONNX, TensorFlow, and PyTorch-bin weights. Existing directories from another source/revision are rejected. Interrupted downloads from the same marked revision may be resumed. Network access is needed for downloads; inference scripts subsequently use local files.

Exact generation settings, seeds, and model revisions are in the experiment commands and [data-source instructions](#data-and-model-sources). Stage timing should be compared on equivalent hardware and runtime; it is not an end-to-end service-latency measurement.

## Complete inference workflow

All commands here are issued from the repository root. Scripts execute the experiment commands with `work/reproduction/` as their working directory; the printed plan shows that directory explicitly. Source datasets and model weights are obtained separately.

### Prepare resources

1. Create the empty workspace with `python3 scripts/setup_workspace.py`.
2. Follow [data-source instructions](#data-and-model-sources) to download the source resources on the inference server and link them with `scripts/link_resources.py`. Run `scripts/verify_resources.py` to check source fingerprints and directory layout.
3. Install the historical GPU environment and fetch pinned model snapshots using [environment instructions](#two-execution-environments).

Workspace setup installs both `supplement/` and the historical `revision_20260909/` runner layout. No manual code-directory copy or `KG_PROJECT_ROOT` variable is needed. Model paths are created by the model-download helper.

### Inspect and run the ordered commands

```sh
.venv-inference/bin/python scripts/run_experiments.py --workspace work/reproduction
```

This **prints commands only**. Execute the complete plan after resource and environment checks:

```sh
.venv-inference/bin/python scripts/run_experiments.py --workspace work/reproduction --run
```

Alternatively, execute the following stages in order with `--stage STAGE --run`:

| Stage | Jobs and purpose |
|---|---|
| `preprocess` | Verify the seven MetaQA source files; create the processed graph/QA view and preprocessing manifest |
| `tuning` | Development-only grid: 50 questions per hop, seed 42, budgets 128/256/512, lambda .3/.5/.7/.9, cost exponent 0/.5/1 |
| `metaqa` | Qwen primary budgets 128/256/512 at seed 20260818; two further B256 samples at seeds 20260819 and 20260820 |
| `components` | Derive the 900-question sampling manifest; preflight and then run all five clean component/label variants; generate the historical component diagnostics used by the relation-only comparison |
| `phi` | Aligned Phi reselection (seed 20260818, B256); then Phi reading of the fixed Qwen-selected text |
| `webqsp` | Verify source layout, audit graph coverage, and run the initial 500-question experiment |
| `shared-smoke` | Artificial two-path graph, exact labeled-token checks, invalid-code scoring, E5 encoding, and a short Qwen generation |
| `shared-webqsp` | Full frozen common-proposal experiment; retain the legacy500 and extension1077 groups |

The plan has 19 commands. `--job NAME` selects one printed job, for example `--job qwen-b256-s20260818 --run`. Each command's output is written to `work/reproduction/logs/NAME.log`. Existing output files are rejected before an experiment is launched. A component preflight does not generate answers; the following `clean-components` job performs generation.

The primary inference runner also records Direct and Oracle diagnostics alongside the four selectors. The primary statistical comparisons and paper plots use the prespecified selector families. The development grid is separate from test inference; fixed selected settings are encoded in the plan. Do not choose a new setting using test scores and call it the same protocol.

### Interrupted runs

For the original runners, preserve any partial output and restart the failed job in a separate output/workspace rather than overwrite it. Already completed jobs can be omitted with `--stage` or `--job`.

The shared-pool runner supports validated resume. From the repository root, use the same interpreter and files:

```sh
.venv-inference/bin/python work/reproduction/revision_20260909/src/run_shared_pool_webqsp.py --project work/reproduction --protocol work/reproduction/revision_20260909/README.md --output work/reproduction/supplement/results/shared_pool_webqsp_all1577_B256.json --batch-size 64 --resume
```

Resume checks source, model, protocol, dependency, and input fingerprints. It does not replace completed predictions. The original six-hour limit was an engineering checkpoint limit, not an outcome-based stopping rule. Continue incomplete runs before analysis; analyses require completed inputs.

### Preserve the experimental distinctions

- MetaQA budgets count the selected evidence body with the selector's tokenizer, not the whole prompt or generation.
- Clean relation/cost/novelty interventions and raw-label interventions share the full B256 reference, but form different statistical families.
- Phi reselection uses Phi's token budget. Fixed Qwen evidence preserves Qwen-selected text, whose Phi token count can exceed the Qwen cap; this is measured by the audit.
- The initial WebQSP design and the common-proposal follow-up are distinct. The latter uses a common pool of at most 1,000 paths, exact code-inclusive B256 packing, a 32-token generation limit, and cyclic condition order.
- Follow `supplement/README.md` for the frozen common-pool design. The primary new group has 1,077 questions; the earlier 500 remain descriptive.

After inference, switch to the CPU environment and follow [analysis instructions](#analysis-from-local-experiment-outputs). A new model run may differ numerically across GPU/runtime combinations; the analysis accepts valid new predictions instead of requiring a specific historical answer.

## Analysis from local experiment outputs

This repository contains code, not saved experimental data. Complete the [inference workflow](#complete-inference-workflow), or supply compatible local outputs with the filenames below. Commands are run from the repository root using the CPU environment in [environment instructions](#two-execution-environments).

### Required inputs inside `work/reproduction/`

| Local path | Origin |
|---|---|
| `results/qwen3_awq_test300_B128.json` | Primary Qwen, B128, seed 20260818 |
| `results/qwen3_awq_test300_B256.json` | Primary Qwen, B256, seed 20260818 |
| `results/qwen3_awq_test300_B512.json` | Primary Qwen, B512, seed 20260818 |
| `results/qwen3_awq_test300_B256_seed20260819.json` | Additional Qwen sample |
| `results/qwen3_awq_test300_B256_seed20260820.json` | Additional Qwen sample |
| `results/clean_component_ablation_qwen_B256.json` | Five clean component/label interventions |
| `results/week2_components_qwen_test300_B256.json` | Historical component diagnostics |
| `results/phi4mini_test300_B256_seed20260818.json` | Aligned Phi reselection |
| `results/week2_shared_evidence_phi_qwenpaths_test300_B256.json` | Phi on fixed Qwen evidence |
| `results/week2_webqsp_qwen_test500_B256.json` | Initial WebQSP experiment |
| `supplement/results/shared_pool_webqsp_all1577_B256.json` | Common-pool follow-up |
| `data/WebQSP.test.json` | Official question/parse file, linked by `link_resources.py` |

### Execute the complete analysis

```sh
.venv-analysis/bin/python scripts/run_analysis.py --workspace work/reproduction
```

The ordered stages are also available individually through `--stage statistics`, `--stage derived`, `--stage audit`, `--stage review-controls`, and `--stage figures`:

1. **Statistics:** eleven JSON/CSV analyses covering the three budgets, two extra sampling seeds, aligned Phi, clean components and labels, template clusters, original WebQSP strata, cross-model contrasts, and fixed-evidence comparisons.
2. **Derived:** joint nine-test primary MetaQA Holm family, separate shared-pool WebQSP family, corrected historical scoring, reader increments, and exploratory cluster intervals.
3. **Audit:** prepare local inputs and compute component-stage contrasts, output-interface diagnostics, official-rule WebQSP sensitivity, and Phi fixed-text token counts.
4. **Review controls:** fixed voting, endpoint filtering, direct stage-effect differences, path-order decomposition, coverage, and generation diagnostics from saved outputs.
5. **Figures:** seven main figure designs and the supplementary component-stage figure, exported under `work/reproduction/manuscript/figures/`.

The additional-sample statistical seeds are 20260819/20260820. Aligned Phi uses inference seed 20260818 but statistical seed 20260819. Cross-model contrasts also use statistical seed 20260819; fixed-evidence contrasts use 20260818. These match the recorded analyses. Core resampling draws and the other seeds are explicit in `scripts/build_statistics.py`.

New statistical files are computed from the supplied predictions; they are not treated as independent frozen reference results. Existing statistics and audit inputs are not overwritten. If one stage fails, inspect its logs, preserve its existing outputs, and rerun the needed later stage or individual command after resolving the cause.

### Statistical families and scoring

Three MetaQA baselines across three budgets form one nine-test Holm family assembled in `derived/revision_analysis.json`. Do not report the per-budget adjustment as that joint family. The three common-pool WebQSP contrasts form a separate family. Cluster, component, raw-label, answer-stage, and scoring-rule analyses retain their specified confirmatory/exploratory scopes. Extra sampling seeds overlap and are not pooled as independent trials. Historical500 and extension1077 remain separate groups.

Raw-record hashes, paired question IDs, source-score consistency, completion status, and token caps are checked by the respective analyzers. A recorded answer is not a test oracle for a newly generated answer. `--verify-original-examples` enables additional checks only when using the original archived predictions; it is off by default. Changing that flag does not change score definitions, sample selection, resampling, or numerical analysis.

For exact independent comparison against separately retained original statistics, the original `rebuild_statistics.sh` and `verify_rebuilt_statistics.py` remain available. They require those reference files locally and are not part of the fresh-run default.

### Figures and outputs

The manuscript uses `study_framework`, `budget_tradeoffs`, `components`, `shared_pool`, `reader_profile`, and `revision_controls`; the additional two designs remain available for analysis. The supplement uses `component_stages`. English is the default. For Chinese figure export, set `KG_FIGURE_LANGUAGES=en,zh` and `KG_CJK_FONT` to an installed CJK font file before the figure stage.

Statistical inputs and outputs, source datasets, PDFs, PNGs, and local logs remain under ignored workspace paths. Only source code and documentation are published.

## Additional review controls

After the primary and clean-component outputs exist, run:

```sh
.venv-analysis/bin/python scripts/run_analysis.py --workspace work/reproduction --stage review-controls
.venv-analysis/bin/python analysis/gen_revision_controls.py --analysis-dir work/reproduction/review_controls --output work/reproduction/manuscript/figures
```

This uses `analysis/revision_controls.py` with the three primary MetaQA budget
files, the clean-component file, the shared WebQSP output, and official WebQSP
parses. The output directory must be empty. These analyses require no new model
calls and use seed 20261110 with 20,000 paired stratified bootstrap samples.
Original scores are checked before computing endpoint-filtering increments,
fixed path-vote comparisons, and the paired difference between reader and
endpoint component effects. Max-vote returns all maximum-count ties;
vote-top5 returns up to five entities, with first-occurrence ties. Five follows
the RoG voting comparison, not test-set optimization.

The complete analysis plan includes this stage and exports precise means to
`review_controls/full_experiment_metrics.csv`. To export other original run
files without averaging across runs:

```sh
.venv-analysis/bin/python analysis/export_full_metrics.py --input run-name=path/to/run.json --output metrics.csv
```

Each `--input` is a descriptive label followed by `=` and an original runner
JSON path; repeat it for multiple files. Existing result files should be
preserved. The manuscript's 286-row table includes an additional historical
stochastic-decoding diagnostic, which is not rerun by the 19-job primary plan.

For saved records supplied separately to reviewers, an alternative is:

```sh
.venv-analysis/bin/python analysis/revision_controls.py --records-dir /path/to/review_revision/records --webqsp-official /path/to/inputs/webqsp_parses.json --output /path/to/new_analysis
```

No such records are stored in this repository. The CPU analysis describes
observed path-order subgroups. The following separate inference workflow tests
matched interfaces, common ordering, generation constraints, and semantic
selection; it preserves the primary outputs.

### Matched reader and semantic controls

Create the workspace from this source snapshot and complete the primary
experiments first. The new scripts use their saved path lists and validate
model, tokenizer, source, and candidate fingerprints. Obtain the official
WebQSP training file and training neighborhoods as described above, then check:

```sh
.venv-inference/bin/python scripts/verify_resources.py --workspace work/reproduction --scope webqsp --include-training
.venv-inference/bin/python scripts/run_review_inference.py --workspace work/reproduction
.venv-inference/bin/python scripts/run_review_inference.py --workspace work/reproduction --run
```

The first plan command prints the four jobs. `--run` executes them sequentially
and records logs under `work/reproduction/logs/review-*.log`. The helper links
the completed common-pool output to the historical path expected by the new
runners, without copying or changing the data. If two different files occupy
those paths, it stops. Select an individual job with `--job reader`,
`--job prepare-semantic`, `--job semantic`, or `--job cost-decoder`. After an interruption, append
`--resume` to one named job with exactly the same source and inputs; preserve
its output folder. GPU inference requires an idle device and obtains an
exclusive lock. Each job has an eight-hour engineering timeout and retains
completed records; timeout is not a scientific stopping rule.

The reader job writes 18,708 condition records. At MetaQA B256 it compares the
original configuration, a no-cost selector, matched labeled-name and labeled-code
prompts, canonical order for both selectors, a 192-token output cap, separate
removal of repetition penalty and trigram blocking, removal of both, and both
labeled interfaces without those constraints. B128 and B512 compare original
and 192-token caps. On the 1,077 new WebQSP questions, RACPS and E5-rank compare
32- and 64-token caps. Original configurations are regenerated in the same
batch8 implementation. Only identical current-run prompt-token IDs and decoding
parameters may share a generated output; historical answers are never used as
new predictions.

The selected members remain fixed in the interface and order probes. Added
code labels or reordered text can increase the evidence length beyond the
original selection cap. Every such count is recorded; no path is silently
removed. The matched labeled-name/code pair shares the annotated evidence and
common instruction prefix. Its output instruction changes. This is distinct
from an equal-token-budget retrieval comparison. Canonical order is
lexicographic order of the complete original path strings; duplicate paths
are retained. The analysis includes the subset where both reordered contexts
still fit the cap.

Semantic preparation retains 2,077 candidate pools: 100 official WebQSP
training questions, 1,077 new WebQSP test questions, and 900 primary MetaQA
questions. The development sample has 50 questions from each official chain
stratum, selected with seeds 20261111/20261112 after graph/topic eligibility
checks. E5 evaluates rank and four density rules: max or min-max normalization,
with cost exponents 0.5 or 1. The chosen rule maximizes development endpoint
macro F1; exact ties follow rank, max/0.5, max/1, min-max/0.5, min-max/1. The
choice is written before test scoring. It is not selected by test reader F1.
The controls are post-review exploratory work, since the original test results
were already known.

MetaQA E5-rank uses all original exact-hop candidates (cap100,000), with no added
lexical prefilter. WebQSP retains the original lexical proposal pool and checks
its per-question hash. Raw cosine scores, standalone costs, hops, selected
indices, and encoding truncation counts are retained locally. Empty candidate
pools remain in the evaluation. After E5 scoring, the reader stage generates
8,631 condition records: MetaQA E5-rank with matched BM25 and RACPS replays under both original decoding and removal of both repetition constraints, and
WebQSP E5-rank, E5-density, and the development-chosen rule. Identical selected
prompts are cached within that run. Batched timing is not per-request latency.

A second-wave cost-by-decoding control follows the initial reader finding. It
regenerates all four cells (full/no-cost selection × original/no-repetition-constraints)
on the same 900 MetaQA B256 questions: 3,600 condition records, with unchanged
paths, prompts, and 96-token generation caps. The thin driver selects these
conditions in the unchanged reader engine and records both source hashes.
This control was specified after seeing the initial trigram-removal effect;
it is exploratory and was frozen before its own outputs. It estimates the
cost-effect difference between decoding configurations directly.

After the three inference matrices reach `PASS`, compute the independent CPU checks:

```sh
.venv-analysis/bin/python analysis/analyze_reader_interventions.py --input work/reproduction/review_followup/reader_formal --meta-reference work/reproduction/results/qwen3_awq_test300_B256.json --output work/reproduction/review_followup/reader_analysis
.venv-analysis/bin/python analysis/analyze_semantic_controls.py --input work/reproduction/review_followup/semantic_formal --meta-reference work/reproduction/results/qwen3_awq_test300_B256.json --output work/reproduction/review_followup/semantic_analysis
.venv-analysis/bin/python analysis/analyze_cost_decoder_controls.py --input work/reproduction/review_followup/cost_decoder_formal --meta-reference work/reproduction/results/qwen3_awq_test300_B256.json --output work/reproduction/review_followup/cost_decoder_analysis
```

Analysis requires complete runs, reconstructs parsed answers and scores, checks
candidate-array hashes, and reports paired stratified intervals plus MetaQA
template-cluster sensitivity (20,000 draws; seed20261110). Direct paired
interactions test order modulation and interface-by-decoding effects. New
intervals are exploratory and pointwise; the primary Holm families are
unchanged. Output CSV/JSON files include exact means, per-question metrics,
cap-hit counts, token overhead, baseline replay concordance, development choice,
and score/cost/path-length diagnostics. The analyzers accept plain JSONL or
`.jsonl.gz` records. Keep the semantic `arrays/` directory for independent
candidate-score checks. None of these generated records belong in GitHub.

Generate the new data-driven figures and supplementary tables from the verified
analysis directories:

```sh
.venv-analysis/bin/python analysis/gen_model_controls.py --reader-analysis work/reproduction/review_followup/reader_analysis --semantic-analysis work/reproduction/review_followup/semantic_analysis --cost-analysis work/reproduction/review_followup/cost_decoder_analysis --output work/reproduction/manuscript/figures
.venv-analysis/bin/python analysis/export_model_tables.py --reader-analysis work/reproduction/review_followup/reader_analysis --semantic-analysis work/reproduction/review_followup/semantic_analysis --cost-analysis work/reproduction/review_followup/cost_decoder_analysis --output work/reproduction/manuscript/generated_model_tables.tex
```

For Chinese labels, add `--chinese` to `gen_model_controls.py` on a machine with
an installed CJK font. The plot script reads the analysis files; it embeds no
experimental scores. The source-code repository retains these generators but
contains none of their PDF, PNG, or LaTeX output artifacts.



## New-question confirmation and token-level audit

The follow-up confirmation freezes four cells on 300 additional MetaQA questions,
100 per hop. It excludes IDs and normalized question strings in all locally
archived `results/*.json` and `revision*/results/*.json` files. Reproducing the
exact paper sample requires those original outputs or the exclusion list and
sample manifest supplied in the separate review archive. Running with an
incomplete archive can produce a different sample; it is not the paper's exact
confirmation. The public repository contains no experimental outputs.

The final paper has 32,139 follow-up condition records (18,708 reader, 8,631
semantic, 3,600 cost/decoder, and 1,200 confirmation). These outputs are supplied
with the paper as `Supplementary_Records_and_Reanalysis.zip`, not in GitHub.
Its `复核数据与代码/round2/publication_snapshot.json` pins the code version.

**Exact replay of the paper's 300 questions.** Extract Supplementary Archive S2
and set `REVIEW_ROUND2` to its `复核数据与代码/round2` directory. After the normal
data/model/workspace setup above, run:

```sh
python scripts/prepare_confirmation_replay.py --review-dir "$REVIEW_ROUND2" --project work/reproduction
.venv/bin/python src/run_independent_confirmation.py --project work/reproduction --plan work/reproduction/confirmation_plan.json --output work/reproduction/review_round2/confirmation
```

The preparation script checks the complete original plan hash, the sample IDs,
all 3,521 exclusion IDs and normalized strings in the post-run ledger, and zero
sample overlap. It copies only that exact plan and sample manifest into a clean
workspace. The runner then reuses the manifest, rather than requiring the
historical source files or resampling. It generates paths and model outputs
again. It will not overwrite a different existing plan or manifest. The ledger
was reconstructed after the run from the 27 unchanged, hash-verified sources;
it is not presented as a separately preregistered artifact. No model or dataset
is downloaded by this preparation command.

**New-sample extension.** To scan your own complete historical outputs and draw
a new sample, use a different clean output directory. This is distinct from
exact replay. Create a new local plan before selecting questions:

```sh
python - <<'PYPLAN'
import json
from pathlib import Path
p = Path("work/reproduction/extension_plan.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"n_per_hop": 100, "seed": 20261111}, indent=2))
PYPLAN
.venv/bin/python src/run_independent_confirmation.py --project work/reproduction --plan work/reproduction/extension_plan.json --output work/reproduction/review_round2/extension
```

The full scientific freeze in the review archive additionally specifies the
primary estimand, decoder controls, exclusions, and reporting rules. The runner
uses the previously prepared MetaQA splits, graph, and Qwen model. Copy the
current `src` directory into `work/reproduction/src` before running if that
workspace does not yet contain the current helper files. Preserve all four
conditions regardless of the outcome. The primary estimand is the change in
full-minus-no-cost F1 after removing both repetition constraints. Sample-level
independence concerns new question instances, not new templates, model families,
or datasets. The model snapshot, exclusion inputs, paths, prompts, and source
hashes are saved locally before inference. GPU lock and occupancy checks protect
other runs; identical current-run prompts can share generated responses.

Compress completed records without deleting the original, then reanalyze:

```sh
python - <<'PYZIP'
import gzip
from pathlib import Path
p = Path("work/reproduction/review_round2/confirmation/records.jsonl")
p.with_suffix(".jsonl.gz").write_bytes(gzip.compress(p.read_bytes(), mtime=0))
PYZIP
.venv-analysis/bin/python analysis/analyze_independent_confirmation.py --input work/reproduction/review_round2/confirmation --output work/reproduction/review_round2/confirmation_analysis
.venv-analysis/bin/python analysis/analyze_answer_provenance.py --source work/reproduction/review_followup --output work/reproduction/review_round2/provenance
.venv/bin/python analysis/trace_copy_blocking.py --records work/reproduction/review_followup/reader_formal/records.jsonl --model-dir work/reproduction/models/Qwen3-8B-AWQ --output work/reproduction/review_round2/mechanism
```

The token audit loads the tokenizer and installed Transformers processor, not
model weights. It verifies forbidden tokens on saved trajectories and separates
shared-prefix cases from later counterfactual replay events. Endpoint membership
does not identify the causal knowledge source of an answer. Figure/table export
scripts `gen_confirmation_figures.py` and `export_confirmation_tables.py` use the
review archive's named round-two subdirectories, documented in its README.

## Source and licensing

This repository provides the research project's reproduction code. No new license for the original project code has been selected on behalf of the authors. Public visibility does not itself grant an additional software license.

MetaQA, WebQSP, UniK-QA resources, and model checkpoints retain the terms set by their respective providers. Their sources and associated publications are listed in the data-source section above. Original benchmark distributions and model weights are not bundled.

The `supplement/vendor` files are frozen helper snapshots from this same research project, retained because the shared-proposal runner records their hashes. Their paths do not imply that a third-party model or dataset is distributed here. The Microsoft WebQSP evaluator is referenced by its official release URL and hash and is not copied into this repository.
