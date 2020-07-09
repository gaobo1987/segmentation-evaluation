import itertools
import json
from typing import List, Tuple, TypeVar
import unicodedata

from tqdm import tqdm
import numpy as np

from segmt_eval.item import Item

T = TypeVar('T')


def edit_ops(A: List[T], B: List[T]) -> List[Tuple[T, T]]:
    """Collect the edits needed to transform A into B

    Parameters
    ----------
    A, B: lists of same type

    Returns
    -------
    edits: aligned pairs of values from A and B.
    """
    mA, mB = len(A), len(B)
    M = np.zeros((mA + 1, mB + 1), np.uint8)
    M[:, 0] = np.arange(0, mA + 1)
    M[0, :] = np.arange(0, mB + 1)
    BP = np.zeros_like(M)
    BP[:, 0] = 3
    BP[0, :] = 4
    # backpointers:
    # 1 = diagonal - equal
    # 2 = diagonal - substitution
    # 3 = vertical - insertion
    # 4 = horizontal - deletion

    for i in range(1, mA + 1):
        for j in range(1, mB + 1):
            if A[i - 1] == B[j - 1]:
                M[i, j] = M[i - 1, j - 1]
                BP[i, j] = 1
            else:
                costs = [M[i - 1, j], M[i, j - 1], M[i - 1, j - 1]]
                argmin: int = np.argmin(costs)
                if argmin == 0:
                    BP[i, j] = 3
                elif argmin == 1:
                    BP[i, j] = 4
                else:
                    BP[i, j] = 2
                M[i, j] = costs[argmin] + 1
    edits = []
    i, j = mA, mB
    while i > 0 or j > 0:
        move = BP[i, j]
        if move == 0:
            break
        elif move in [1, 2]:
            edits.append((A[i - 1], B[j - 1]))
            i -= 1
            j -= 1
        elif move == 3:
            edits.append((A[i - 1], None))
            i -= 1
        elif move == 4:
            edits.append((None, B[j - 1]))
            j -= 1
    return edits[::-1]


def is_contiguous(items: List[Item]) -> bool:
    """Checks whether all items in a sorted list of items are adjacent to each other
    """
    orig, shift = itertools.tee(items)
    next(shift, None)
    return all(o.endOffSet == s.startOffSet for o, s in zip(orig, shift))


def align_items(a: List[Item], b: List[Item]) -> List[Tuple[List[Item], List[Item]]]:
    """Align items between a and b.

    Parameters
    ----------
    a, b: Lists of items to align. Both lists are assumed to be composed of contiguous non-overlapping items
    and span the same total length

    Returns
    -------
    List of pairs. Each pair contains a list of items from a and a list of items from b, s.t.
    they cover the same span. In the ideal case, i.e. every item is aligned to a single item,
    each list in the pair is a singleton.
    """
    a = sorted(a, key=lambda it: (it.startOffSet, -it.endOffSet))
    b = sorted(b, key=lambda it: (it.startOffSet, -it.endOffSet))

    min_a, max_a = min(item.startOffSet for item in a), max(item.endOffSet for item in a)
    min_b, max_b = min(item.startOffSet for item in b), max(item.endOffSet for item in b)

    if min_a != min_b or max_a != max_b:
        raise ValueError('a and b need to cover the same total span')

    start_ix, end_ix = min_a, max_a

    result = []
    a_ix = b_ix = 0
    char_ix = start_ix
    curr_pair = ([], [])
    while char_ix < end_ix:
        curr_a = a[a_ix]
        curr_b = b[b_ix]
        if char_ix < min(curr_a.startOffSet, curr_b.startOffSet):
            char_ix += 1
            continue
        if curr_a.endOffSet == curr_b.endOffSet:
            # wrap up the alignment
            curr_pair[0].append(curr_a)
            curr_pair[1].append(curr_b)
            result.append(curr_pair)
            curr_pair = ([], [])
            char_ix = curr_a.endOffSet
            a_ix += 1
            b_ix += 1
        elif curr_a.endOffSet < curr_b.endOffSet:
            curr_pair[0].append(curr_a)
            char_ix = curr_a.endOffSet
            a_ix += 1
        else:  # curr_a.endOffSet > curr_b.endOffSet
            curr_pair[1].append(curr_b)
            char_ix = curr_b.endOffSet
            b_ix += 1
    return result


def load_json(data_path):
    data_file = open(data_path, 'r', encoding='utf8')
    data_str = data_file.read()
    data_file.close()
    return json.loads(data_str)


def save_json(data_path, data):
    data_file = open(data_path, 'w', encoding='utf-8')
    json.dump(data, data_file, ensure_ascii=False)
    data_file.close()


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def strip_accents_and_lower_case(s: str) -> str:
    return strip_accents(s).lower()


def relax_str(s: str) -> str:
    return strip_accents_and_lower_case(s).strip()


def find_gold_term(query_gold: list, item_text: str, item_start_char_index: int):
    match = {}
    for item in query_gold:
        start_offset = int(item['startOffSet'])
        if item['item'] == item_text and start_offset == item_start_char_index:
            match = item
            break
    return match


def compose_evaluation_data(segmenter, gold_path, save_path):
    if segmenter is not None:
        eval_data = load_json(gold_path)
        for q in tqdm(eval_data):
            target_query = q['expanded_query'] if 'expanded_query' in q else q['query']
            output = segmenter.segment(target_query)
            q['pred'] = json.loads(output)
        save_json(save_path, eval_data)


def convert_items_to_bio(items: List[Item]):
    if not items:
        return []

    items = sorted(items, key=lambda it: (it.startOffSet, -it.endOffSet))
    tags = []
    i = 0
    while i < len(items):
        if items[i].ner == '':
            tags.append('O')
            i += 1
        else:
            ner = items[i].ner[0]['ner']
            tags.append(f'B-{ner}')
            start, end = items[i].startOffSet, items[i].endOffSet
            i += 1
            while i < len(items) and items[i].endOffSet <= end:
                if items[i].startOffSet != start:
                    tags.append(f'I-{ner}')
                i += 1
    return tags
