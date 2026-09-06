from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from erc_engine.domain.models import DatasetKind, RuntimeSettings


def find_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "datasets").is_dir():
            return candidate
    raise RuntimeError("Cannot find a repository containing datasets/.")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class SettingsFactory:
    """Abstract Factory: constructs a coherent dataset-specific object graph configuration."""

    @staticmethod
    def create(dataset: DatasetKind, start: Path | None = None) -> RuntimeSettings:
        root = find_root((start or Path.cwd()).resolve())
        load_env_file(root / ".env")
        dataset_dir = _path(root, os.getenv("DATASETS_DIR", "datasets"))
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-") or "model"
        upper = dataset.value.upper()
        data_default = dataset_dir / f"{dataset.value}_coggate_utterance_inputs.csv"
        data_path = _path(root, os.getenv(f"{upper}_DATA_PATH", str(data_default)))
        output_dir = root / "outputs" / dataset.value
        output_path = _path(
            root,
            os.getenv(
                f"{upper}_OUTPUT_PATH",
                str(output_dir / f"{dataset.value}_erc_all_agents_equal_{slug}_{date}.csv"),
            ),
        )
        prior_default = root / "proposed" / upper / "prototype_prior" / "outputs" / (
            "appraisal_prior_weights_8d_optimized_dev_gpt-5.4-nano.json"
            if dataset is DatasetKind.IEMOCAP
            else "appraisal_prior_weights_dev_gpt-5.4-nano.json"
        )
        return RuntimeSettings(
            root=root,
            dataset=dataset,
            data_path=data_path,
            output_path=output_path,
            sample_output_path=output_path.with_name(f"{output_path.stem}_samples.csv"),
            summary_output_path=output_path.with_name(f"{output_path.stem}_summary.json"),
            prior_weights_path=_path(root, os.getenv(f"{upper}_PRIOR_WEIGHTS_PATH", str(prior_default))),
            model=model,
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            limit_dialogues=int(os.getenv("LIMIT_DIALOGUES", "0")),
            conversation_workers=int(os.getenv("MAX_WORKERS", "32" if dataset is DatasetKind.IEMOCAP else "40")),
            agent_workers=int(os.getenv("MAX_AGENT_WORKERS", "6")),
            baseline_batch_size=int(os.getenv("BASELINE_BATCH_SIZE", "25")),
            write_every=int(os.getenv("WRITE_EVERY", "1")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            use_learned_prior=_flag("USE_LEARNED_PRIOR", dataset is DatasetKind.MELD),
        )

