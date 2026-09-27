from abc import ABC, abstractmethod
from src.ai.agents.base_agent import BaseAgent

class BaseRetriever(ABC):
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve relevant context chunks for a given query."""
        pass

    def build_index(self, chunks: list[str]) -> None:
        """Optional: Build a local index or sync to remote. Default is no-op."""
        pass

    @property
    @abstractmethod
    def has_index(self) -> bool:
        """Check if the retriever has an initialized index/collection."""
        pass
