from typing import List, Dict
from segmt_eval.item import Item


class TaskMetric:
    def single(self, a: List[Item], b: List[Item]) -> Dict[str, float]:
        raise NotImplementedError

    def aggregate(self) -> Dict[str, float]:
        raise NotImplementedError
