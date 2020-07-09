from typing import List, Dict

import sklearn.metrics

from segmt_eval.item import Item
from segmt_eval.utils import align_items, edit_ops

from .base import TaskMetric

__all__ = ['LemmaMetric']


class LemmaMetric(TaskMetric):
    def __init__(self, mode: str, skip_unaligned: bool = True, **kwargs):
        self.mode = mode
        self.skip_unaligned = skip_unaligned

        self._a_lemmas = []
        self._b_lemmas = []

    def single(self, a: List[Item], b: List[Item]) -> Dict[str, float]:
        a = [it for it in a if it.isMinimumToken]
        b = [it for it in b if it.isMinimumToken]
        alignment = align_items(a, b)
        a_lemmas, b_lemmas = [], []
        for items_a, items_b in alignment:
            if len(items_a) == len(items_b):
                a_lemmas.append(items_a[0].lemma)
                b_lemmas.append(items_b[0].lemma)
            else:
                if self.skip_unaligned:
                    continue
                edits = edit_ops([it.lemma for it in items_a],
                                 [it.lemma for it in items_b])
                for lemma_a, lemma_b in edits:
                    a_lemmas.append('MISALIGNED' if lemma_a is None else lemma_a)
                    b_lemmas.append('MISALIGNED' if lemma_b is None else lemma_b)
        self._a_lemmas.extend(a_lemmas)
        self._b_lemmas.extend(b_lemmas)
        return self._score(a_lemmas, b_lemmas)

    def aggregate(self) -> Dict[str, float]:
        return self._score(self._a_lemmas, self._b_lemmas)

    def _score(self, a_lemmas, b_lemmas):
        if self.mode == 'reference':
            return {
                'accuracy': sum(1 for a, b in zip(a_lemmas, b_lemmas) if a == b) / len(a_lemmas)
            }
        elif self.mode == 'agreement':
            return {
                'kappa': sklearn.metrics.cohen_kappa_score(a_lemmas, b_lemmas)
            }
