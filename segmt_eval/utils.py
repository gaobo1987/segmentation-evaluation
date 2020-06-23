import json
from tqdm import tqdm
import unicodedata


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


def convert_items_to_bio(items):
    if items == []:
        return []
    if 'startOffSet' in items[0]:
        start_key = 'startOffSet'
    elif 'startOffset' in items[0]:
        start_key = 'startOffset'
    else:
        raise ValueError('no startOffSet key in items')
    if 'endOffSet' in items[0]:
        end_key = 'endOffSet'
    elif 'endOffset' in items[0]:
        end_key = 'endOffset'
    else:
        raise ValueError('no endOffset key in items')

    items = sorted(items, key=lambda it: (it[start_key], -it[end_key]))
    tags = []
    i = 0
    while i < len(items):
        if items[i]['ner'] == '':
            tags.append('O')
            i += 1
            continue
        ner = items[i]['ner'][0]['ner']
        tags.append(f'B-{ner}')
        start, end = items[i][start_key], items[i][end_key]
        i += 1
        while i < len(items) and items[i][end_key] <= end:
            if items[i][start_key] != start:
                tags.append(f'I-{ner}')
            i += 1
    return tags
