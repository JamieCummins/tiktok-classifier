# TikTok Mental Health Classifier

Two-pass LLM pipeline that classifies TikTok videos as mental-health-related
based on text metadata (description, transcript, suggested words).

This is the third stage of the TikTok data pipeline:
`tiktok-donations-backend` → `tiktok-scraper` → `tiktok-training-data` → **`tiktok-classifier`**.
Input metadata comes from `../pipeline-data/02-metadata/tiktok_metadata.parquet`
(written by `tiktok-scraper`); ground-truth labels come from
`../pipeline-data/03-handcoded/handcoded_examples.csv` (maintained live by
`tiktok-training-data`'s labelling app). Neither file is stored in this repo.

- **Pass 1 — screen** (`--stage screen`): liberal first pass over all videos.
  Casts a wide net; false positives are acceptable.
  Writes `is_mental_health_broad_<model>` to the parquet.
- **Pass 2 — classify** (`--stage classify`): fine-grained second pass over
  screen-passing rows only. Requires genuine engagement with a recognised
  mental health condition.
  Writes `is_mental_health_<model>` and optionally `is_mental_health_<model>_rationale`.

Both stages support atomic checkpointing: safe to interrupt and resubmit —
already-classified rows are skipped on resume.

## For collaborators

### Where to start

The classification logic lives in two files:

- **`prompts.py`** — the methodological core. Contains both system prompts with their
  full TRUE/FALSE criteria, how video metadata is formatted for the model, and how
  few-shot examples are constructed. Start here if you want to review or improve the
  classification criteria.
- **`classify.py`** — the implementation. `classify_one_vllm()` shows how structured
  decoding is used to constrain outputs; `parse_response()` handles output parsing.
  `TEMPERATURE`, `RATIONALE_MAX_TOKENS`, `LABELS`, and `TEXT_COLS` at the top of the
  file are the key parameters.

Everything else (`hpc/`, `prepare_imhi_eval.py`, the batch loop in `classify()`) is
infrastructure and can be ignored on a first read.

### Testing a change locally

The fastest feedback loop is Ollama + the held-out test set:

```bash
uv sync
ollama pull gemma4:27b-it-qat   # or any instruct model

# Optionally edit prompts.py, then:
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet \
  --stage classify --evaluate ../pipeline-data/03-handcoded/handcoded_examples.csv \
  --filter is_superficial_mental_health_handcoded=TRUE \
  --rationale --backend ollama -v
```

This classifies the ~50 held-out rows in `handcoded_examples.csv` (use=test),
prints accuracy and a confusion matrix, and lists mismatches with rationales so
you can see exactly where the model disagrees with the ground truth. Results are
appended to `experiment_log.csv` for comparison across runs.

### Key design decisions

- **Two-pass rather than one**: the screen pass uses a high-recall prompt to avoid
  losing edge cases; the classify pass applies stricter criteria only to the ~8% of
  videos that pass. This keeps the expensive classify prompt away from the bulk of data.
- **Model label suffix on output columns** (e.g. `is_mental_health_gemma4_26b`): allows
  results from multiple models to coexist in the same parquet for direct comparison.
- **Structured/guided decoding**: the model is constrained at the token level to
  produce only valid output — `TRUE`/`FALSE` for label-only mode, or a JSON object
  with a hard-constrained `label` field and a free `rationale` string. See the
  [Structured and guided decoding](#structured-and-guided-decoding) section for details.

## Setup

### Local (requires Python 3.11+ and [uv](https://docs.astral.sh/uv/))

```bash
uv sync
ollama pull gemma4:e4b   # or any other instruct model
```

### HPC (Imperial CX3)

Run once on the login node (internet access required, no GPU needed):

```bash
bash hpc/setup_hpc_env.sh
```

This creates the `tiktok` conda environment, installs dependencies from
`pyproject.toml`, and downloads the model weights specified in `model.env`
to `$HOME/huggingface`. To switch models, update `model.env` and rerun
the setup script to download the new weights.

## Running the classifier

### Local

```bash
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet --stage screen --concurrency 4 -v
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet --stage classify --rationale --concurrency 4 -v
```

The model name, model label, and the classify-pass `--filter` are all derived
from `model.env` automatically — no extra flags needed.

### HPC (PBS)

```bash
qsub hpc/run_smoke_test.pbs               # 20-row sanity check
qsub -v ROWS=1000 hpc/run_smoke_test.pbs  # scale test
qsub hpc/run_classify.pbs                 # full dataset; resubmit if walltime hit
```

Model configuration (name, label, quantization, thinking mode) lives in
`model.env` at the repo root — edit that file to switch models. The PBS
scripts source it automatically.

### Hosted (OpenRouter or Jev)

No GPU or local server needed. Export the API key in your shell, then pick the
backend:

```bash
export OPENROUTER_API_KEY=...   # https://openrouter.ai/keys
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet --stage screen --backend openrouter --concurrency 32 -v
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet --stage classify --backend openrouter --concurrency 32 -v

export TYPESAFE_API_KEY=...     # https://console.typesafe.ai/keys
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet --stage screen --backend jev --concurrency 32 -v
```

- **`openrouter`** runs the same generative prompts against a hosted model. It
  reads `OPENROUTER_MODEL` / `OPENROUTER_MODEL_LABEL` from `model.env` (default
  `qwen/qwen3.6-27b`, the model used on the HPC). Results go in their own
  columns because the provider's precision may differ from our FP8 weights.
- **`jev`** sends the metadata to [Jev](https://docs.typesafe.ai/), TypeSafe's
  discriminative (non-generative) model. The metadata fields go in as a JSON
  `state`, and the stage prompt's criteria become the instructions of one
  true/false question (`to_jev_instructions()` in `prompts.py`). Jev returns a
  calibrated p(TRUE) rather than text, so the label is `p >= --threshold`
  (default 0.5). `--rationale` is not available. It reads `JEV_MODEL` /
  `JEV_MODEL_LABEL` (pinned to `jev-1.14.0`, because the `jev-latest` alias
  moves between versions).

Hosted runs retry rate limits and transient errors with backoff. A row that
still fails is left empty and picked up on the next run; bad API keys or model
names stop the run at once. A token count (and OpenRouter's reported cost) is
printed at the end.

## Structured and guided decoding

This is the most finicky part of the pipeline.

### Why structured decoding?

Without constraints, LLMs can produce verbose output ("The answer is TRUE because…")
or refuse to answer with a valid label at all. Post-hoc string matching is fragile.
Structured/guided decoding solves this at the token level: the model physically cannot
emit tokens that would violate the constraint. A `TRUE`/`FALSE` label is always returned.

### The four cases

The approach to enforcing structured output varies depending on the backend, and whether we are prompting for a rationale. The `--rationale` flag is used exclusively for debugging - by having the model report its reasoning for a certain label, we were able to iterate on the prompts. Rationale is not used in production. 

| Backend | Mode | Mechanism | Where in code |
|---------|------|-----------|---------------|
| vLLM | label only | `structured_outputs: {choice: ["TRUE","FALSE"]}` (XGrammar) | `classify_one_vllm` |
| vLLM | `--rationale` | `guided_json: RATIONALE_SCHEMA` | `classify_one_vllm` |
| Ollama | label only | free generation + `parse_label` fallback | `classify_one_ollama` |
| Ollama | `--rationale` | `format: RATIONALE_SCHEMA` (native Ollama) | `classify_one_ollama` |
| OpenRouter | label only | free generation + `parse_label` fallback | `classify_one_openrouter` |
| OpenRouter | `--rationale` | strict `response_format` JSON schema, routed only to providers that support it | `classify_one_openrouter` |
| Jev | label only | not generative: returns p(TRUE), labelled by `--threshold` | `classify_one_jev` |

Ollama's and OpenRouter's label-only modes do not use structured decoding (neither
exposes a choice constraint), but in practice the models we use follow
the "Reply with a single token" instruction reliably on label-only prompts. Any
stray output shows up as `UNKNOWN`.

### p(TRUE) scores

In label-only mode, vLLM and OpenRouter requests also ask for the top logprobs
of the first output token. `_prob_true()` adds up every alternative that could
start each label ("TRUE", "True", "TR", ...) and renormalises over the two
labels. The result goes in `prob_true_broad_<model>` (screen) or
`prob_true_<model>` (classify). Jev fills the same columns with
its own probability. Evaluation reports ROC AUC from these scores, which shows
how well the model ranks videos regardless of where the cutoff sits.

### RATIONALE_SCHEMA

For rationale mode on both backends, the model is constrained to produce a JSON object
matching this schema (defined as `RATIONALE_SCHEMA` in `classify.py`):

```json
{
  "type": "object",
  "properties": {
    "label":     {"type": "string", "enum": ["TRUE", "FALSE"]},
    "rationale": {"type": "string"}
  },
  "required": ["label", "rationale"]
}
```

`label` is hard-constrained to `TRUE`/`FALSE`; `rationale` is a free string.
`parse_response()` JSON-decodes the output and extracts both fields, falling back to
line-based parsing only if JSON decoding fails (which it shouldn't when constraints
are active).

### vLLM legacy API fallback

vLLM has two guided-decoding backends:
- **XGrammar** (newer): `extra_body: {structured_outputs: {choice: [...]}}` — used by default
- **Outlines** (older): `extra_body: {guided_choice: [...]}` — used as fallback

The first request attempts XGrammar; if it raises an error mentioning `structured_outputs`,
`use_legacy[0]` is set to `True` and all subsequent requests use `guided_choice`.
This means there's no manual configuration needed when switching vLLM versions.

### Disabling thinking mode (Qwen3)

Qwen3 models include a chain-of-thought "thinking" mode that emits a `<think>…</think>`
block before the answer. This contaminates structured-output parsing — the grammar tries
to match the thinking block against the schema and fails.

Thinking mode must be disabled at **two** levels:
1. **Server level**: `--default-chat-template-kwargs '{"enable_thinking": false}'` in the
   `vllm serve` command (see `hpc/config.sh`)
2. **Per-request level**: `extra_body: {chat_template_kwargs: {enable_thinking: false}}`
   sent with every request (the `no_think` dict in `classify_one_vllm`)

Both are required. The server-level flag alone is unreliable on some vLLM versions
(tracked in [vLLM issue #35574](https://github.com/vllm-project/vllm/issues/35574));
the per-request flag alone has no effect without the server-level flag. The double-layer
approach is the only reliable solution we found.

Non-Qwen models (Gemma, Llama, etc.) are unaffected — the `enable_thinking` key is
ignored if the model's chat template doesn't recognise it.

## Evaluation

`handcoded_examples.csv` contains hand-labelled videos:

| Column | Description |
|--------|-------------|
| `is_mental_health_handcoded` | Ground truth for classify stage (`TRUE`/`FALSE`) |
| `is_superficial_mental_health_handcoded` | Ground truth for screen stage |
| `use` | `train` (few-shot example) or `test` (held-out evaluation) |
| `rationale` | Explanation included in few-shot prompts for the classify stage |

```bash
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet \
  --stage classify \
  --evaluate ../pipeline-data/03-handcoded/handcoded_examples.csv \
  --filter is_superficial_mental_health_handcoded=TRUE \
  --rationale --concurrency 2 -v
```

Prints accuracy, confusion matrix, mismatches, and ROC AUC when p(TRUE) scores are
available. Full results are saved to `eval_results_classify.csv`. Metrics are
appended to `experiment_log.csv`.

Add `--backend openrouter` or `--backend jev` (dropping `--rationale` for Jev)
to compare hosted models on the same held-out rows.

### End-to-end (both passes)

The per-stage evaluations score each pass in isolation: the classify evaluation
only sees rows a human marked as passing the screen, so videos the model's
screen wrongly drops never count against it. `--stage pipeline` scores what the
pipeline actually produces:

```bash
python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet \
  --stage pipeline \
  --evaluate ../pipeline-data/03-handcoded/handcoded_examples.csv -v
```

It runs the screen on the test rows, then the classify pass on those the screen
let through. A row the screen rejects is final `FALSE`; otherwise it takes the
classify label. The final label is scored against `is_mental_health_handcoded`,
and the errors are split into mental-health videos lost at the screen (listed
by URL) and errors made by the classify pass. `--rationale` applies to the
classify pass only. Results go to `eval_results_pipeline.csv` and the run is
logged with `stage=pipeline`.

### IMHI benchmark

`prepare_imhi_eval.py` converts the [IMHI dataset](https://huggingface.co/datasets/Tianlin668/IMHI)
test split into the evaluation format for cross-dataset comparison:

```bash
pip install datasets
python prepare_imhi_eval.py   # writes imhi_eval.csv

python classify.py ../pipeline-data/02-metadata/tiktok_metadata.parquet \
  --stage classify --evaluate imhi_eval.csv --backend vllm
```

## CLI reference

### classify.py

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | required | `screen`, `classify`, or `pipeline` (with `--evaluate` only) |
| `--backend` | `ollama` | `ollama`, `vllm`, `openrouter` or `jev` |
| `--endpoint` | backend-specific | Server URL (model is set via `model.env`) |
| `--model` | `model.env` | Override the model name |
| `--model-label` | `model.env` | Override the output-column suffix |
| `--threshold P` | `0.5` | `jev` only: label TRUE when p(TRUE) >= P |
| `--evaluate CSV` | — | Evaluate against ground-truth CSV; prints accuracy + confusion matrix |
| `--filter COL=VAL` | — | Only classify rows where column matches value |
| `--rationale` | off | Request a one-sentence explanation alongside each label |
| `--examples CSV` | — | Inject `use=train` rows from CSV as few-shot examples |
| `--concurrency N` | `4` | Parallel requests |
| `--batch_size N` | `200` | Rows per checkpoint |
| `--timeout N` | `120` | Per-request timeout in seconds |
| `--output-col NAME` | stage-specific | Override output column name |
| `--cols COL ...` | `description transcript suggested_words` | Metadata fields to include |
| `-v` | off | Verbose: print each result as it arrives |

## Output columns

Column names are suffixed with the model label to support multi-model comparison:

| Column | Stage | Values |
|--------|-------|--------|
| `is_mental_health_broad_<model>` | screen | `TRUE`, `FALSE`, `SKIPPED`, `UNKNOWN`, `NO_TEXT`, `TOO_LONG` |
| `is_mental_health_<model>` | classify | `TRUE`, `FALSE`, `SKIPPED`, `UNKNOWN`, `NO_TEXT`, `TOO_LONG` |
| `is_mental_health_<model>_rationale` | classify + `--rationale` | one-sentence explanation |
| `prob_true_broad_<model>` | screen | p(TRUE) where the backend provides it (not Ollama, not `--rationale`) |
| `prob_true_<model>` | classify | as above |

`NO_TEXT` marks rows with no description, transcript or suggested words; no
request is sent for them.
