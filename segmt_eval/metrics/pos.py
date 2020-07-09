from typing import List, Dict

import sklearn.metrics

from segmt_eval.item import Item
from segmt_eval.utils import align_items, edit_ops

from .base import TaskMetric

__all__ = ['POSMetric']


class POSMetric(TaskMetric):
    def __init__(self, mode: str, average='micro', skip_unaligned: bool = True, **kwargs):
        self.mode = mode
        self.average = average
        self.skip_unaligned = skip_unaligned

        self._a_postags = []
        self._b_postags = []

    def single(self, a: List[Item], b: List[Item]) -> Dict[str, float]:
        a = [it for it in a if it.isMinimumToken]
        b = [it for it in b if it.isMinimumToken]
        alignment = align_items(a, b)
        a_postags, b_postags = [], []
        for items_a, items_b in alignment:
            if len(items_a) == len(items_b) == 1:
                a_postags.append(items_a[0].pos)
                b_postags.append(items_b[0].pos)
            else:  # unaligned items
                if self.skip_unaligned:
                    continue
                edits = edit_ops([it.pos for it in items_a],
                                 [it.pos for it in items_b])
                for postag_a, postag_b in edits:
                    a_postags.append('MISALIGNED' if postag_a is None else postag_a)
                    b_postags.append('MISALIGNED' if postag_b is None else postag_b)
        score = self._score(a_postags, b_postags)
        self._a_postags.extend(a_postags)
        self._b_postags.extend(b_postags)
        return score

    def _score(self, a_pos, b_pos) -> Dict[str, float]:
        if self.mode == 'reference':
            prec, rec, f1, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    a_pos, b_pos, average=self.average, zero_division=0
                )
            return {
                'precision': prec,
                'recall': rec,
                'fscore': f1
            }
        elif self.mode == 'agreement':
            return {
                'kappa': sklearn.metrics.cohen_kappa_score(a_pos, b_pos)
            }

    def aggregate(self) -> Dict[str, float]:
        return self._score(self._a_postags, self._b_postags)
