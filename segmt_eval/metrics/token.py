from typing import List, Dict
from collections import Counter
from operator import itemgetter

from segmt_eval.item import Item
import numpy as np

from .base import TaskMetric

__all__ = ['TokenMetric']


class EditCounter(dict):
    _keys = {'n_match', 'n_ad', 'n_trans', 'w_trans',
             'n_pot_bounds', 'n_bounds_A',
             'n_bounds_B'}

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            if key in self._keys:
                return 0
            else:
                raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

    def __add__(self, other):
        if not isinstance(other, EditCounter):
            raise ValueError(f'cannot add EditCounter and {other.__class__}')

        return EditCounter(
            n_match=self.n_match + other.n_match,
            n_ad=self.n_ad + other.n_ad,
            n_trans=self.n_trans + other.n_trans,
            w_trans=self.w_trans + other.w_trans,
            n_pot_bounds=self.n_pot_bounds + other.pot_bounds,
            n_bounds_A=self.n_bounds_A + other.n_bounds_A,
            n_bounds_B=self.n_bounds_B + other.n_bounds_B
        )

    def __iadd__(self, other):
        if not isinstance(other, EditCounter):
            raise ValueError(f'cannot add EditCounter and {other.__class__}')
        self.n_match += other.n_match
        self.n_ad += other.n_ad
        self.n_trans += other.n_trans
        self.w_trans += other.w_trans
        self.n_pot_bounds += other.n_pot_bounds
        self.n_bounds_A += other.n_bounds_A
        self.n_bounds_B += other.n_bounds_B
        return self


class TokenMetric(TaskMetric):
    def __new__(cls, mode: str, **kwargs):
        if mode == 'agreement':
            return TokenAgreementMetric(**kwargs)
        else:
            return TokenReferenceMetric(**kwargs)


class TokenReferenceMetric(TaskMetric):
    def __init__(self, **kwargs):
        self._counter = Counter({'correct': 0, 'n_gold': 0, 'n_pred': 0})

    def single(self, a: List[Item], b: List[Item]) -> Dict[str, float]:
        a_ix = b_ix = 0
        counter = Counter({'correct': 0, 'n_gold': len(a), 'n_pred': len(b)})
        while a_ix < len(a) and b_ix < len(b):
            if a[a_ix].startOffSet < b[b_ix].startOffSet:
                a_ix += 1
            elif a[a_ix].startOffSet > b[b_ix].startOffSet:
                b_ix += 1
            else:
                if a[a_ix].endOffSet == b[b_ix].endOffSet:
                    counter['correct'] += 1
                a_ix += 1
                b_ix += 1
        self._counter += counter
        return TokenReferenceMetric._score(counter)

    def aggregate(self) -> Dict[str, float]:
        return TokenReferenceMetric._score(self._counter)

    @staticmethod
    def _score(counter: Dict[str, int]) -> Dict[str, float]:
        correct, n_gold, n_pred = itemgetter('correct', 'n_gold', 'n_pred')(counter)
        return {
            'precision': correct / n_pred if n_pred else 0.,
            'recall': correct / n_gold if n_gold else 0.,
            'fscore': 2 * correct / (n_pred + n_gold) if n_pred + n_gold else 0.
        }


class TokenAgreementMetric(TaskMetric):
    def __init__(self, **kwargs):
        self._edit_counts = EditCounter()

    def single(self, a: List[Item], b: List[Item]) -> Dict[str, float]:
        a = [it for it in a if it.isMinimumToken]
        b = [it for it in b if it.isMinimumToken]
        boundaries_a = TokenAgreementMetric._boundary_array(a)
        boundaries_b = TokenAgreementMetric._boundary_array(b)
        edit_counts = TokenAgreementMetric._count_edits(boundaries_a, boundaries_b)
        score = TokenAgreementMetric._boundary_edit_kappa(edit_counts)
        self._edit_counts += edit_counts
        return score

    def aggregate(self) -> Dict[str, float]:
        return TokenAgreementMetric._boundary_edit_kappa(self._edit_counts)

    @staticmethod
    def _boundary_edit_kappa(edit_counts: EditCounter) -> Dict[str, float]:
        """Calculate the boundary agreement based on a set of edits

        Parameters
        ----------
        edit_counts: counts of edits

        Returns
        -------
        Cohen's kappa on the boundary agreement
        """

        def score(n_match, n_ad, n_trans, w_trans):
            return 1 - (n_ad + w_trans) / (n_ad + n_trans + n_match)

        observed = score(edit_counts.n_match, edit_counts.n_ad, edit_counts.n_trans, edit_counts.w_trans)
        chance = edit_counts.n_bounds_A * edit_counts.n_bounds_B / edit_counts.n_pot_bounds ** 2
        return {'boundary_edit_kappa': (observed - chance) / (1 - chance)}

    @staticmethod
    def _count_edits(ba1: np.ndarray, ba2: np.ndarray, winlen: int = 1) -> EditCounter:
        """Count the edits needed to align the boundaries in ba1 and ba2

        Parameters
        ----------
        ba1, ba2: boolean array marking boundaries. should have same shape
        winlen: size of the window for transpositions

        Returns
        -------
        EditCounter

        """
        subs = ba1 ^ ba2
        trans = []
        i = 0
        while i < subs.shape[0]:
            if not subs[i]:
                i += 1
                continue
            for offs in range(1, winlen + 1):
                if i + offs >= subs.shape[0]:
                    break
                if subs[i + offs] and ((ba1[i] and ba2[i + offs]) or (ba2[i] and ba1[i + offs])):
                    trans.append(1 - offs / (winlen + 1))
                    i += offs + 1
                    break
            i += 1
        return EditCounter(
            n_match=(ba1 & ba2).sum(),
            n_ad=subs.sum() - len(trans) * 2,
            n_trans=len(trans),
            w_trans=sum(trans),
            n_pot_bounds=len(ba1),
            n_bounds_A=ba1.sum(),
            n_bounds_B=ba2.sum()
        )

    @staticmethod
    def _boundary_array(items: List[Item]) -> np.ndarray:
        """Convert an item list into an array of boundary marks

        Parameters
        ----------
        items: list of items

        Returns
        -------
        boolean array of length of number of characters - 1. each true value marks a boundary
        """
        # get masses
        prev_end = 0
        masses = []
        for it in items:
            start, end = it.startOffSet, it.endOffSet

            if start > prev_end:
                masses.append(start - prev_end)
            masses.append(end - start)
            prev_end = end
        arr = np.zeros((sum(masses) - 1,), dtype=np.bool)
        pos = 0
        for mass in masses:
            curr = pos + mass - 1
            if curr < sum(masses) - 1:
                arr[curr] = 1
            pos += mass
        return arr
