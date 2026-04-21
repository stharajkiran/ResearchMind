from dataclasses import dataclass

@dataclass
class RetrieverMetrics:
    semantic_found: float = 0.0
    technical_found: float = 0.0
    semantic_count: int = 0
    technical_count: int = 0

    @property
    def semantic_recall(self) -> float:
        return (
            self.semantic_found / self.semantic_count
            if self.semantic_count > 0
            else 0.0
        )

    @property
    def technical_recall(self) -> float:
        return (
            self.technical_found / self.technical_count
            if self.technical_count > 0
            else 0.0
        )

    def update_found(self, q: dict, found: bool) -> None:
        if q["category"] == "semantic":
            self.semantic_found += found
            self.semantic_count += 1
        elif q["category"] == "technical":
            self.technical_found += found
            self.technical_count += 1