# Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models

[![CI](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullah-x-bd/unlearning/actions/workflows/ci.yml)

This repository is the experimental artifact for a ground-up reconstruction of **Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models**. The paper is downstream of the artifact: the old manuscript supplies hypotheses, not evidence, and manuscript claims are written only from frozen experiment outputs.

## Research question

Can language-model training be engineered so that a later data-deletion request becomes a reproducible counterfactual computation rather than an ad hoc post-training edit?

The exact target is a **trace-preserving deletion counterfactual**. The execution plan fixes microbatch slots, RNG seeds, optimizer-step boundaries and learning-rate values. Forgotten records contribute zero loss while the original trace remains fixed. Ordinary densely repacked retraining is measured separately because it changes the computation.

## Research design

The rebuilt project has three deliberately different empirical studies.

### Study A: controlled systems and scaling

Pythia + WikiText-103 is the systems study. It measures exactness, model scale, deletion geometry, deterministic execution, checkpoint spacing, provenance cost, physical redaction and replay latency without changing model family at every scale.

Primary models:

- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b` as a budget-dependent extension

Human-readable configs use Pythia `step143000`; release execution resolves it to a full 40-character Hugging Face commit SHA through the artifact lock.

### Study B: first-class standardized TOFU benchmark

The standardized benchmark uses **Llama 3.2 1B Instruct + TOFU** and a pinned OpenUnlearning checkout. The TOFU exporter reproduces OpenUnlearning's Llama chat template and stores `labels.npy` so training loss is restricted to the final assistant response rather than prompt/system tokens.

Splits:

- `forget01` / `retain99`
- `forget05` / `retain95`
- `forget10` / `retain90`

Publication approximate baselines come from OpenUnlearning, not our local reference implementations:

- GradAscent
- GradDiff
- NPO
- SimNPO

The same pinned OpenUnlearning evaluator is used for standardized benchmark metrics. Direct comparison tables must use the same chosen target checkpoint and retain reference.

### Study C: MUSE external validation

MUSE is a larger, budget-dependent extension after Studies A and B. The planned track covers News and Books, removal-size scaling and sequential-deletion sustainability. Both the OpenUnlearning integration and original MUSE repository are pinned by commit.

See [`docs/EXTERNAL_BENCHMARKS.md`](docs/EXTERNAL_BENCHMARKS.md).

## Pinned external research code

External repositories are not copied into this project. `scripts/bootstrap_upstreams.py` fetches detached checkouts at commits recorded in `external/upstreams.lock.yaml`:

- OpenUnlearning: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- original MUSE implementation: `6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e`
- Pythia reference repository: `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`

## Artifact freezing

`locks/artifact-sources.yaml` enumerates remote model/dataset refs and prepared datasets. Before any publication run:

```bash
python scripts/bootstrap_upstreams.py
python scripts/freeze_artifacts.py
python scripts/verify_release_lock.py
```

The generated `locks/artifacts.lock.json` contains full Hugging Face commit SHAs, exact external Git commits, every prepared-dataset file SHA-256 and a directory digest. Release runners refuse to execute if the source manifest changes, an upstream checkout moves or a prepared dataset no longer matches its lock.

The final `artifacts.lock.json` is generated **after** dataset preparation and then frozen with the release evidence.

## Exact replay core

Implemented mechanisms include:

- immutable microbatch execution plans
- deterministic per-microbatch RNG seeds
- fixed accumulation boundaries
- float32 learning-rate values in the trace
- 32-byte WAL records with CRC32
- segment SHA-256 verification
- ordered-ID manifest with SHA-256/HMAC support
- exact model-state and optimizer-state hashing
- checkpoint restore and replay
- independent trace-preserving deletion oracle
- slot-preserving masked replay
- physical-row filtering replay
- repacked retain-only counterfactual
- physically materialized redacted token stores
- optional benchmark label masks
- provenance-field ablations
- checkpoint/replay tradeoff sweeps
- exact XOR rollback patches
- cohort-scoped LoRA experiments
- approximate curvature hot path

## WAL scope

Each binary WAL record is exactly 32 bytes:

```text
uint64 ordered_sample_ids_digest
uint64 microbatch_seed
float32 learning_rate
uint32 logical_optimizer_step
uint16 microbatch_length
uint16 flags
uint32 crc32
```

The ordered IDs are stored separately. Every run reports WAL bytes, manifest bytes and total provenance cost. The project never treats 32 bytes as total provenance storage.

## Exactness evidence

Every exact comparison records model SHA-256, optimizer SHA-256, exact-hash equality, unequal tensors/elements, maximum absolute difference, L2 difference, applied/skipped updates, runtime, checkpoint size and replay distance. Byte-exact claims require both state hashes and tensor comparison to agree.

## Prepare Study A data

```bash
python -m pip install -e '.[llm,dev]'
python scripts/prepare_dataset.py \
  --dataset Salesforce/wikitext \
  --subset wikitext-103-raw-v1 \
  --model EleutherAI/pythia-160m \
  --output data/prepared/wikitext103-pythia-256 \
  --max-records 20000 \
  --sequence-length 256
```

## Prepare Study B data

First freeze the remote refs so the full Llama and TOFU revisions are known. Then use those exact SHAs with:

```bash
python scripts/prepare_tofu_openunlearning.py \
  --output data/prepared/tofu-llama32-1b-openunlearning \
  --model-revision <FULL_LLAMA_SHA> \
  --dataset-revision <FULL_TOFU_SHA> \
  --max-length 512
```

The exporter writes immutable IDs, input IDs, attention masks, answer-only labels, split-ID files and per-file hashes.

## Software verification

```bash
python -m pip install -e '.[dev]'
pytest
uas core-smoke --output runs/core-smoke
```

The tiny CI model is software testing only and never scientific evidence.

## Publication release runs

Do not use an unlocked config for publication evidence. After the final data lock exists:

```bash
python scripts/run_release.py configs/pythia-160m.yaml
python scripts/run_release_matrix.py configs/matrix-main.yaml
python scripts/run_release.py configs/benchmarks/tofu-llama32-1b-trace.yaml
```

The release scripts replace the readable model revision with the exact SHA stored in `locks/artifacts.lock.json`.

## Official OpenUnlearning evaluation

Fetch the pinned checkout and install its environment according to the upstream repository. Then evaluate one of our exported/local model checkpoints with:

```bash
python scripts/openunlearning_adapter.py tofu-eval \
  --checkpoint <MODEL_CHECKPOINT_DIRECTORY> \
  --forget-split forget10
```

Run publication-authoritative approximate baselines with:

```bash
python scripts/openunlearning_adapter.py tofu-baselines \
  --forget-split forget10 \
  --target-checkpoint <SAME_TARGET_CHECKPOINT>
```

The adapter stores the upstream commit and exact commands in machine-readable manifests.

Our local `src/unlearning_at_scale/baselines.py` and `scripts/run_approximate_baselines.py` remain **sanity/reference implementations only**.

## MUSE extension

After the Pythia and TOFU 1B studies are complete:

```bash
python scripts/openunlearning_adapter.py muse --data-split News
python scripts/openunlearning_adapter.py muse --data-split Books
```

MUSE is explicitly excluded from the initial low-budget run.

## Scientific claim discipline

Failures are results. Distributed exactness is not claimed until a dedicated multi-GPU study exists. Approximate methods are never called exact because a behavioral audit passes. Legal erasure implications are kept distinct from technical state-equivalence claims.

The full sequence is in [`docs/RUNBOOK.md`](docs/RUNBOOK.md), candidate claims in [`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md), baseline policy in [`docs/BASELINES.md`](docs/BASELINES.md), and reproducibility requirements in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Status

**Core experimental framework:** implemented

**Pinned external benchmark architecture:** implemented

**Artifact-freeze/preflight layer:** implemented

**Multi-model paid GPU results:** not yet run

**Paper rewrite:** intentionally deferred until results are frozen

## License

Apache-2.0.
