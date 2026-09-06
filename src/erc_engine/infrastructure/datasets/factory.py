from erc_engine.domain.models import DatasetKind

from .iemocap import IemocapAdapter
from .meld import MeldAdapter


class DatasetAdapterFactory:
    @staticmethod
    def create(kind: DatasetKind, data_path):
        if kind is DatasetKind.IEMOCAP:
            return IemocapAdapter(data_path)
        if kind is DatasetKind.MELD:
            return MeldAdapter()
        raise ValueError(f"Unsupported dataset: {kind}")

