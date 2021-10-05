from typing import Any, Optional
from dataclasses import dataclass


@dataclass(order=False)
class Claim:
    source: Any
    text: str
    score: Optional[float] = None
    question: Optional[str] = None

    def __lt__(self, other):
        if isinstance(other, Claim):
            return self.text < other.text
        return True

    def __gt__(self, other):
        if isinstance(other, Claim):
            return self.text > other.text
        return False
