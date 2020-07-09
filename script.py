from typing import List, Dict

from segmt_eval.evaluator import Evaluator
from segmt_eval.item import Item


import json
import os

data_file = os.path.join('example_data', 'eval.json')
gold, pred = zip(*[(sent['gold'], sent['pred'])
                   for sent in json.load(open(data_file))])

def itemize(data: List[List[Dict]]) -> List[List[Item]]:
    r = []
    for sent in data:
        s = []
        for values in sent:
            if not 'isStopWord' in values:
                values['isStopWord'] = False
            s.append(Item(**values))
        r.append(s)
    return r
    #return [[Item(isStopWord=False, **d) for d in sent] for sent in data]


gold = itemize(gold)
pred = itemize(pred)
evaluator = Evaluator(tasks=['pos', 'token', 'lemma'], mode='agreement')
print(evaluator.evaluate(gold, pred))

evaluator = Evaluator(tasks=['pos', 'token', 'lemma', 'ner'], mode='reference', average='micro', verbose=True)
print(evaluator.evaluate(gold, pred))