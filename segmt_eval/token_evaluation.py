from typing import List, Dict
import numpy as np


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
            n_match = self.n_match + other.n_match,
            n_ad = self.n_ad + other.n_ad,
            n_trans = self.n_trans + other.n_trans,
            w_trans = self.w_trans + other.w_trans,
            n_pot_bounds = self.n_pot_bounds + other.pot_bounds,
            n_bounds_A = self.n_bounds_A + other.n_bounds_A,
            n_bounds_B = self.n_bounds_B + other.n_bounds_B
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


def calculate_boundary_edits(A: List[dict], B: List[dict]) -> EditCounter:
    """Calculate edits needed to align boundaries of A and B

    Parameters
    ----------
    A
    B

    Returns
    -------
    Cohen's kappa on boundary agreement
    """
    def get_boundary_array(items) -> np.ndarray:
        # get masses
        prev_end = 0
        masses = []
        for it in items:
            start, end = it['startOffSet'], it['endOffSet']

            if start > prev_end:
                masses.append(start - prev_end)
            masses.append(end - start)
            prev_end = end
        arr = np.zeros((sum(masses) - 1, ), dtype=np.bool)
        pos = 0
        for mass in masses:
            curr = pos + mass - 1
            if curr < sum(masses) - 1:
                arr[curr] = 1
            pos += mass
        return arr

    def count_edits(ba1, ba2, winlen=1) -> Dict[str, int]:
        subs = ba1 ^ ba2
        trans = []
        i = 0
        while i < subs.shape[0]:
            if not subs[i]:
                i += 1
                continue
            for offs in range(1, winlen+1):
                if i + offs >= subs.shape[0]:
                    break
                if subs[i + offs] and ((ba1[i] and ba2[i+offs]) or (ba2[i] and ba1[i+offs])):
                    trans.append(1 - offs / (winlen+1))
                    i += offs + 1
                    break
            i += 1
        return EditCounter(
            n_match = (ba1 & ba2).sum(),
            n_ad = subs.sum() - len(trans) * 2,
            n_trans = len(trans),
            w_trans = sum(trans),
            n_pot_bounds = len(ba1),
            n_bounds_A = ba1.sum(),
            n_bounds_B = ba2.sum()
        )

    # calculate agreement score
    bA, bB = get_boundary_array(A), get_boundary_array(B)
    assert len(bA) == len(bB), f'sentences do not align \n{A}\n{B}'
    return count_edits(bA, bB)


def boundary_edit_score(edit_counts: EditCounter) -> float:
    """

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
    return (observed - chance) / (1 - chance)


def evaluate_token(A: List[List[dict]], B: List[List[dict]], keep_intermediate=False) -> Dict:
    """Evaluate the agreement on tokenization between two data sets

    Parameters
    ----------
    A: list of list of items
    B: list of list of items

    Returns
    -------
    Cohen's kappa on the aggregated boundary edit distance
    """
    edits = EditCounter()
    if keep_intermediate:
        scores = []
    for a, b in zip(A, B):
        curr_edits = calculate_boundary_edits([item for item in a if item['isMinimumToken']],
                                          [item for item in b if item['isMinimumToken']])
        if keep_intermediate:
            scores.append(boundary_edit_score(curr_edits))
        edits += curr_edits

    result = {'boundary agreement': boundary_edit_score(edits)}
    if keep_intermediate:
        result['scores'] = scores
    return result