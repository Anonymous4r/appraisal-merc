from erc_engine.infrastructure.datasets import DatasetAdapterFactory
from erc_engine.infrastructure.llm import OpenAIJsonGateway
from erc_engine.infrastructure.persistence.csv_repository import CsvResultRepository
from erc_engine.strategies.prior import PriorFactory
from erc_engine.strategies.prompts import CognitivePromptStrategy
from erc_engine.strategies.voting import EqualAgentWeightedVoting

from .pipeline import EmotionRecognitionPipeline


class PipelineFactory:
    """Composition Root: the only place allowed to know concrete implementations."""

    @staticmethod
    def create(settings):
        dataset = DatasetAdapterFactory.create(settings.dataset, settings.data_path)
        prompts = CognitivePromptStrategy(dataset.labels, settings.dataset)
        model = OpenAIJsonGateway(settings, prompts.system_prompt)
        prior = PriorFactory.create(settings.prior_weights_path, dataset.labels, settings.use_learned_prior)
        voting = EqualAgentWeightedVoting(dataset.labels, dataset.normalize_label)
        pipeline = EmotionRecognitionPipeline(settings, dataset, model, prompts, prior, voting)
        repository = CsvResultRepository(settings.output_path, settings)
        return dataset, pipeline, repository

