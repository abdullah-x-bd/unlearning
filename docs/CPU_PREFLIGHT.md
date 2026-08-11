# CPU publication preflight

The non-GPU preflight is a hard gate before paid model training. It runs on a clean GitHub-hosted Linux runner and performs the following sequence:

1. install the frozen research runtime
2. run the complete regression suite
3. run the exact-replay core smoke test
4. fetch OpenUnlearning, MUSE and Pythia at pinned Git commits
5. resolve all Hugging Face refs to full 40-character commits
6. prepare the 20,000-record WikiText trace corpus plus 256 synthetic canary records
7. prepare all 4,000 TOFU examples with the pinned OpenUnlearning Llama 3.2 tokenizer and answer-only supervision
8. validate array shapes, unique IDs, manifest hashes, label integrity and TOFU forget/retain partitions
9. freeze all models, datasets, prepared files and upstream repositories into `locks/artifacts.lock.json`
10. verify the resulting lock from disk
11. record the Python/package environment
12. upload the prepared token stores as a workflow artifact

## Verified data

### WikiText trace study

- 20,000 source records
- 256 canary records
- 20,256 total records
- sequence length 256
- pinned WikiText commit `b08601e04326c79dfdd32d625aee71d232d685c3`
- pinned Pythia-160M tokenizer/model commit `b56d9bee36300031aeea723b73c4d62ac7fa71a2`
- prepared directory digest `e423aa197acf0f38b374e8829b823673cb97d53170cfb3af8a97b3e858f0d47d`

### TOFU standardized study

- 4,000 total records
- fixed sequence length 512
- forget01: 40, retain99: 3,960
- forget05: 200, retain95: 3,800
- forget10: 400, retain90: 3,600
- every forget/retain pair is disjoint and exactly partitions the full 4,000-row set
- every record has at least one supervised assistant-response token
- supervised labels equal the corresponding input tokens and never fall on padding
- minimum target tokens per example: 7
- mean target tokens per example: 34.82625
- maximum target tokens per example: 92
- pinned TOFU commit `324592d84ae4f482ac7249b9285c2ecdb53e3a68`
- pinned OpenUnlearning Llama 3.2 1B full-model tokenizer commit `88e31200b97e4c0c04ae0d2f0b591f427046d192`
- prepared directory digest `e1704f7f7073b892fe57ff423455f842b78e60ff03a4ee007505f6880f79360c`

## Frozen software baseline

- Python 3.11
- PyTorch 2.4.1
- Transformers 4.51.3
- Datasets 3.0.1
- Hugging Face Hub 0.36.0
- Accelerate 0.34.2
- NumPy 2.2.3

These versions deliberately align the benchmark-facing runtime with the pinned OpenUnlearning environment. The first attempted preflight exposed that unconstrained dependencies had drifted to Transformers 5.15 and PyTorch 2.13, changing chat-template behavior. The dependency ranges were therefore replaced with the frozen research versions before the successful preflight.

## Evidence files

- `locks/artifacts.lock.json`
- `locks/preparation-resolved.json`
- `locks/cpu-preflight-validation.json`
- `locks/cpu-preflight-environment.json`
- `locks/cpu-preflight-pip-freeze.txt`

The generated token arrays remain outside Git because they are experimental data artifacts. The CPU workflow uploads them as `cpu-preflight-release-bundle`; a GPU environment can either consume that bundle or regenerate the arrays from the committed sources and must pass `scripts/verify_release_lock.py` before a release run is allowed.
