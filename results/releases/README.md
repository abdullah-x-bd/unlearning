# Frozen empirical release records

This directory contains compact, human-readable release records for the publication-facing experiments. Each record points back to the immutable workflow runs, artifact identifiers, hashes, environment metadata, and machine-readable summaries used to support the claims ledger.

| Release record | Model / benchmark | Primary result |
| --- | --- | --- |
| [`pythia-160m-2026-08-11/`](pythia-160m-2026-08-11/README.md) | Pythia 160M + WikiText trace corpus | exact identity replay and exact trace-preserving deletion replay across four deletion geometries; physical redaction verified |
| [`pythia-2.8b-2026-08-11/`](pythia-2.8b-2026-08-11/README.md) | Pythia 2.8B + WikiText trace corpus | exact identity replay and exact random-5% trace-preserving deletion replay at 2.8B scale; physical redaction verified |
| [`tofu-llama32-1b-forget10-2026-08-11/`](tofu-llama32-1b-forget10-2026-08-11/README.md) | Llama 3.2 1B Instruct + TOFU `forget10` | exact identity replay and exact physically redacted trace-preserving deletion replay across 1,498,482,688 model elements plus optimizer state |
| [`tofu-openunlearning-eval-2026-08-12/`](tofu-openunlearning-eval-2026-08-12/README.md) | OpenUnlearning TOFU evaluation of the frozen Llama states | standardized behavioral evaluation of the hash-verified original and exact-deletion checkpoints with durable metrics and artifact provenance |

## Evidence policy

- Values in these records are copied from successful machine-generated evidence artifacts, not reconstructed manually from paper tables.
- Exactness claims require model and optimizer state equality under the pinned release environment.
- Behavioral metrics are reported separately from state-equality claims.
- The official OpenUnlearning `retain90` checkpoint is treated as a calibration reference, not silently relabeled as a matched local retain-only retrain.
- Approximate baseline superiority, multi-GPU exactness, arbitrary-stack exactness, privacy guarantees, and legal compliance are not inferred unless their dedicated evidence exists.

The current claim boundary is maintained in [`../../docs/CLAIMS_LEDGER.md`](../../docs/CLAIMS_LEDGER.md).
