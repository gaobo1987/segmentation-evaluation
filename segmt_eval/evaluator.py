from typing import List, Dict

from tqdm import tqdm

from segmt_eval.metrics import TokenMetric, POSMetric, LemmaMetric, NERMetric

__all__ = ['Evaluator']

from .item import Item

METRICS = {
    'token': TokenMetric,
    'lemma': LemmaMetric,
    'pos': POSMetric,
    'ner': NERMetric
}


class Evaluator:
    def __init__(self, tasks: List[str], mode: str = 'reference', verbose: bool = False, **kwargs):
        """

        Parameters
        ----------
        tasks: list of tasks to evaluate. Tasks are `token`, `lemma`, `pos`, `ner`
        mode: `reference` or `agreement`. Reference evaluates a predicted set against a ground truth set.
            Agreement evaluates two predicted sets against each other.
        verbose: display progress bar
        kwargs: keyword arguments specific to each task
        """
        self.tasks = tasks
        self.mode = mode
        self.verbose = verbose
        self.kwargs = kwargs

    def evaluate(self, A: List[List[Item]], B: List[List[Item]]) -> Dict[str, Dict[str, float]]:
        """Evaluate B against A.


        Parameters
        ----------
        A: List of list of Items. In reference mode, this would be the gold set.
        B: List of list of Items. In reference mode, this would be the predicted set.

        Returns
        -------
        dictionary from tasks to score names to scores.
        """
        metrics = {
            task: METRICS[task](self.mode, **self.kwargs)
            for task in self.tasks
        }
        for a, b in tqdm(zip(A, B), disable=not self.verbose):
            for metric in metrics.values():
                metric.single(a, b)
        return {
            task: metric.aggregate() for task, metric in metrics.items()
        }
