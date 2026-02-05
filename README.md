# VILLAIN at AVerImaTeC: Verifying Image–Text Claims via Multi-Agent Collaboration
[![arXiv](https://img.shields.io/badge/arXiv-2602.04587-b31b1b.svg?style=flat)](https://arxiv.org/abs/2602.04587)

This repository contains the code for 🦹🏼VILLAIN, **the first-place** system in the AVerImaTeC shared task.

The system description paper will be published in the proceedings of the Ninth FEVER Workshop (co-located with EACL 2026). [[paper]](https://arxiv.org/abs/2602.04587)

## Prepare Data

Please refer to [DATASET.md](DATASET.md) for detailed instructions on preparing the dataset.

## Quick Start

```bash
# 1. Setup environment variables
cp .env.pub .env
# Edit .env with your API keys

# 2. Run the pipeline (uses scripts/cfg/default.yaml)
./scripts/run_multi_agent_pipeline.sh

# 3. Evaluate results
./scripts/eval_multi_agent_pipeline.sh
```

# License & Attribution
The code is shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0).
