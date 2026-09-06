# ERC Engine

The two research notebooks are represented as a runnable Python package with a shared staged pipeline for IEMOCAP and MELD.

## Architecture

```text
CLI -> Settings Factory -> Composition Root
                           |-> Dataset Adapter (Template Method)
                           |-> Prompt Strategy
                           |-> JSON Model Gateway (retry + telemetry)
                           |-> Assignment Policy
                           |-> Six parallel expert workers
                           |-> Appraisal Prior Strategy
                           |-> Voting Strategy
                           `-> CSV Repository -> Evaluation Query Service
```

The package applies Ports & Adapters, Dependency Inversion, Abstract Factory, Strategy, Policy Object, Template Method, Repository, Unit of Work, Facade and Composition Root patterns. Dataset-specific behavior is restricted to adapters and settings; orchestration is shared.

## Commands

```bash
python -m pip install -e .
python -m erc_engine inspect iemocap
python -m erc_engine inspect meld
python -m erc_engine run iemocap --limit 1
python -m erc_engine evaluate iemocap
```

`inspect` never calls the model API. `run` requires `OPENAI_API_KEY`. Paths are loaded from `.env`; `DATASETS_DIR` points to the local CSV directory.

