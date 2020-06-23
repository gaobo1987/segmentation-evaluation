import os
from tqdm import tqdm
from segmt_eval.confusion_matrix import ConfusionMatrix
from segmt_eval.token_evaluation import evaluate_token
from segmt_eval.utils import load_json, find_gold_term, convert_items_to_bio, relax_str


def evaluate_ner(gold: list, pred: list, labels: list):
    from segmt_eval.ner_evaluation.ner_eval import Evaluator
    evaluator = Evaluator(gold, pred, labels)
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

    return results, results_per_label


# Evaluate based on predefined json datasets
# on the performance of 'pos', 'lemma' or 'ner'
# strict == False: prediction and gold values are lower-cased, stripped of accents and margin spaces
# strict == True: strict comparison between prediction and gold
def evaluate(data, key: str, strict=False):
    if isinstance(data, str) and os.path.isfile(data) and data[-4:] == 'json':
        data = load_json(data)
    elif not isinstance(data, list):
        raise Exception("The data input is neither a json file path nor a list.")
    if key == 'pos' or key == 'lemma':
        cm = ConfusionMatrix()
        for q in tqdm(data):
            for t in q['pred']:
                if t['isMinimumToken']:
                    pred = relax_str(t[key]) if not strict else t[key]
                    gold_term = find_gold_term(q['gold'], t['item'], int(t['startOffSet']))
                    gold = gold_term.get(key, None)
                    gold = relax_str(gold) if not strict and gold is not None else gold
                    cm.add_value_to_cell(pred, gold, 1)
        result = {
            'weighted_f1': cm.wgt_f1(),
            'weighted_precision': cm.wgt_precision(),
            'weighted_recall': cm.wgt_recall(),
            'average_f1': cm.avg_f1(),
            'average_precision': cm.avg_precision(),
            'average_recall': cm.avg_recall(),
            'accuracy': cm.accuracy()
        }
        if key == 'pos':
            result.update({
                'f1s': cm.f1s(),
                'precisions': cm.precisions(),
                'recalls': cm.recalls()
            })
        return result
    elif key == 'ner':
        gold_labels = []
        pred_labels = []
        label_set = set()
        for q in tqdm(data):
            # construct bio labels
            sent_gold_labels = convert_items_to_bio(q['gold'])
            sent_pred_labels = convert_items_to_bio(q['pred'])
            if len(sent_pred_labels) == len(sent_gold_labels):
                pass
            elif len(sent_pred_labels) > len(sent_gold_labels):
                diff = len(sent_pred_labels) - len(sent_gold_labels)
                sent_pred_labels = sent_pred_labels[:-diff]
            elif len(sent_gold_labels) > len(sent_pred_labels):
                diff = len(sent_gold_labels) - len(sent_pred_labels)
                sent_gold_labels = sent_gold_labels[:-diff]
            assert len(sent_pred_labels) == len(sent_gold_labels)

            gold_labels.append(sent_gold_labels)
            pred_labels.append(sent_pred_labels)
            # add to label set
            sent_labels = list(set(sent_gold_labels + sent_pred_labels))
            sent_labels = [lbl.replace('B-', '').replace('I-', '') for lbl in sent_labels]
            label_set.update(sent_labels)

        return evaluate_ner(gold_labels, pred_labels, list(label_set))
    elif key == 'token':
        return evaluate_token(zip(*[(d['gold'], d['pred']) for d in data]))
    else:
        return None
