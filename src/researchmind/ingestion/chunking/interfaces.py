from abc import ABC, abstractmethod

from researchmind.ingestion.models import Chunk, ParsedPaper


class Chunker(ABC):
    """Split a ParsedPaper into indexable Chunk objects."""

    @abstractmethod
    def chunk(self, paper: ParsedPaper) -> list[Chunk]:
        ...
