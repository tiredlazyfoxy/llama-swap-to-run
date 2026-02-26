# LLama-cpp Service

Multi-model LLM orchestration service using [llama-swap](https://github.com/mostlygeek/llama-swap) with CUDA GPU support. Manages a catalog of quantized GGUF models served via llama-cpp-server behind Docker.

## Key Files

- **models.csv** — Source of truth for the model catalog (id, HuggingFace repo, context sizes, embedder flag)
- **config.yaml** — Generated service config consumed by llama-swap; defines model commands, groups, and GPU assignments
- **create.py** — Regenerates `config.yaml` models section from `models.csv` (HuggingFace remote refs)
- **create_locals.py** — Same as `create.py` but generates commands for local `.gguf` files in `/models/`
- **docker-compose.yaml** — Runs llama-swap container (port 9292) with NVIDIA GPU passthrough
- **download_models** — Bash script that downloads models from HuggingFace based on `models.csv`
- **context-sizes.csv** — Reference data for model context window sizes

## Workflow

1. Edit `models.csv` to add/remove models
2. Run `python create.py` (or `create_locals.py` for local files) to regenerate `config.yaml`
3. Run `./download_models` to fetch any new model files from HuggingFace
4. `docker compose up` to start the service

## Tech Stack

Python 3 + PyYAML, Docker with NVIDIA CUDA, llama-cpp-server, HuggingFace Hub CLI
