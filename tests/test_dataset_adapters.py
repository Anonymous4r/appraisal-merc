from pathlib import Path

from erc_engine.domain.models import DatasetKind
from erc_engine.infrastructure.datasets import DatasetAdapterFactory
from erc_engine.infrastructure.settings import SettingsFactory


def test_both_datasets_load_as_conversations():
    for kind in DatasetKind:
        settings = SettingsFactory.create(kind, Path.cwd())
        adapter = DatasetAdapterFactory.create(kind, settings.data_path)
        conversations = adapter.load(settings.data_path, limit=1)
        assert len(conversations) == 1
        assert conversations[0].utterances
        assert all(item.emotion in adapter.labels for item in conversations[0].utterances)

