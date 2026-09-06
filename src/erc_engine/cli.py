from __future__ import annotations

import argparse
import json

from erc_engine.application.evaluation import EvaluationService
from erc_engine.application.factory import PipelineFactory
from erc_engine.application.runner import ConcurrentBatchRunner
from erc_engine.domain.models import DatasetKind
from erc_engine.infrastructure.datasets import DatasetAdapterFactory
from erc_engine.infrastructure.persistence.csv_repository import CsvResultRepository
from erc_engine.infrastructure.settings import SettingsFactory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="erc", description="Run the modular ERC architecture.")
    result.add_argument("command", choices=("inspect", "run", "evaluate"))
    result.add_argument("dataset", choices=tuple(item.value for item in DatasetKind))
    result.add_argument("--limit", type=int, help="Override conversation limit for this process.")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = SettingsFactory.create(DatasetKind(args.dataset))
    if args.limit is not None:
        from dataclasses import replace
        settings = replace(settings, limit_dialogues=max(0, args.limit))

    dataset = DatasetAdapterFactory.create(settings.dataset, settings.data_path)
    conversations = dataset.load(settings.data_path, settings.limit_dialogues)
    if args.command == "inspect":
        print(json.dumps({
            "dataset": settings.dataset.value,
            "data_path": str(settings.data_path),
            "conversations": len(conversations),
            "utterances": sum(len(item.utterances) for item in conversations),
            "labels": dataset.labels,
            "output_path": str(settings.output_path),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        dataset, pipeline, repository = PipelineFactory.create(settings)
        frame = ConcurrentBatchRunner(pipeline, repository, settings.conversation_workers, settings.write_every).run(conversations)
    else:
        repository = CsvResultRepository(settings.output_path, settings)
        frame = repository.read()
    summary = EvaluationService(dataset.labels).evaluate(frame, settings.summary_output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
