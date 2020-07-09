from typing import List, Dict

from segmt_eval.item import Item
from segmt_eval.utils import convert_items_to_bio
from .ner_evaluation.ner_eval import Evaluator as NEREvaluator

from .base import TaskMetric

__all__ = ['NERMetric']


class NERMetric(TaskMetric):
    def __init__(self, mode: str, skip_unaligned: bool = False, **kwargs):
        if mode != 'reference':
            raise ValueError(f'only `reference` mode is supported')
        self.skip_unaligned = skip_unaligned

        self._gold_labels = []
        self._pred_labels = []
        self._label_set = set('O')

    def single(self, a: List[Item], b: List[Item]) -> Dict[str, float]:
        sent_gold_labels = convert_items_to_bio(a)
        sent_pred_labels = convert_items_to_bio(b)
        if len(sent_gold_labels) != len(sent_pred_labels):
            if self.skip_unaligned:
                return {}
            m = min(len(sent_gold_labels), len(sent_pred_labels))
            sent_gold_labels = sent_gold_labels[:m]
            sent_pred_labels = sent_pred_labels[:m]
        sent_labels =  set([label.replace('B-', '').replace('I-', '')
                            for label in sent_gold_labels + sent_pred_labels])
        self._gold_labels.append(sent_gold_labels)
        self._pred_labels.append(sent_pred_labels)
        self._label_set |= sent_labels
        return NERMetric.score([sent_gold_labels], [sent_pred_labels], sent_labels)

    @staticmethod
    def score(a: List[str], b: List[str], tags: List[str]) -> Dict[str, float]:
        evaluator = NEREvaluator(a, b, tags)
        results, results_per_label = evaluator.evaluate()
        for k in results:
            p = results[k]['precision']
            r = results[k]['recall']
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            results[k]['f1'] = f1

        for label in results_per_label:
            for k in results_per_label[label]:
                p = results_per_label[label][k]['precision']
                r = results_per_label[label][k]['recall']
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                results_per_label[label][k]['f1'] = f1
        return {
            'results': results,
            'results_per_label': results_per_label
        }

    def aggregate(self) -> Dict[str, float]:
        return NERMetric.score(self._gold_labels, self._pred_labels, self._label_set)

