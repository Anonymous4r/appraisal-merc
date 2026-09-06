from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm


class ConcurrentBatchRunner:
    """Application Service: resumable outer concurrency with buffered commits."""

    def __init__(self, pipeline, repository, workers: int, write_every: int):
        self.pipeline = pipeline
        self.repository = repository
        self.workers = max(1, workers)
        self.write_every = max(1, write_every)

    def run(self, conversations):
        completed = self.repository.completed_ids()
        pending = [item for item in conversations if item.conversation_id not in completed]
        buffer = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.pipeline.execute, item): item.conversation_id for item in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ERC conversations"):
                buffer.append(future.result())
                if len(buffer) >= self.write_every:
                    self.repository.append(buffer)
                    buffer.clear()
        self.repository.append(buffer)
        return self.repository.read()

