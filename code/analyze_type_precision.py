import json
import os

print("relation", "raw_p", "few_p", "p inc", "raw_type_p", "few_type_p", "t inc", sep='\t')


relations = ['P1001', 'P101', 'P103', 'P106', 'P108', 'P127', 'P1303', 'P131', 'P136', 'P1376', 'P138', 'P140', 'P1412', 'P159', 'P17', 'P176', 'P178', 'P19', 'P190', 'P20', 'P264', 'P27', 'P276', 'P279', 'P30', 'P31', 'P36', 'P361', 'P364', 'P37', 'P39', 'P407', 'P413', 'P449', 'P463', 'P47', 'P495', 'P527', 'P530', 'P740', 'P937']
fewshot_dir_path = 'case_based_10_train'
raw_dir_path = 'lama'

with open('data/type_file/bert.json', 'r') as f:
    relation_token = json.load(f)


def get_type_precision(prediction_list, type_token_set):
    type_correct_cnt = 0
    total = len(prediction_list)
    for prediction in prediction_list:
        top_word = prediction['topk'][0]['token']
        type_correct_cnt += 1 if top_word in type_token_set else 0
    type_presicion = 100.0 * type_correct_cnt / total
    return type_presicion

def get_precision(prediction_list):
    correct_cnt = 0
    total = len(prediction_list)
    for prediction in prediction_list:
        top_word = prediction['topk'][0]['token']
        answer = prediction['obj_label']
        correct_cnt += 1 if top_word == answer else 0
    presicion = 100.0 * correct_cnt / total
    return presicion

for relation in relations:
    with open(os.path.join(fewshot_dir_path, relation, relation + '_predictions.jsonl'), 'r') as f:
        fewshot_prediction_list = f.readlines()
        fewshot_prediction_list = [json.loads(pred) for pred in fewshot_prediction_list]
    with open(os.path.join(raw_dir_path, relation, relation + '_predictions.jsonl'), 'r') as f:
        raw_prediction_list = f.readlines()
        raw_prediction_list = [json.loads(pred) for pred in raw_prediction_list]
    type_token_set = set(relation_token[relation])
    raw_p = get_precision(raw_prediction_list)
    few_p = get_precision(fewshot_prediction_list)
    p_inc = few_p - raw_p
    raw_type_p = get_type_precision(raw_prediction_list, type_token_set)
    few_type_p = get_type_precision(fewshot_prediction_list, type_token_set)
    t_inc = few_type_p - raw_type_p
    print(
        relation,
        raw_p,
        few_p,
        p_inc,
        raw_type_p,
        few_type_p,
        t_inc,
        sep='\t'
    )
