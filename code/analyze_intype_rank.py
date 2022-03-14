import json
import os

print("relation_id", "relation_label", "inc", "unchange", "dec", "inc_per", "unchange_per", "dec_per", sep='\t')


relations = ['P1001', 'P101', 'P103', 'P106', 'P108', 'P127', 'P1303', 'P131', 'P136', 'P1376', 'P138', 'P140', 'P1412', 'P159', 'P17', 'P176', 'P178', 'P19', 'P190', 'P20', 'P264', 'P27', 'P276', 'P279', 'P30', 'P31', 'P36', 'P361', 'P364', 'P37', 'P39', 'P407', 'P413', 'P449', 'P463', 'P47', 'P495', 'P527', 'P530', 'P740', 'P937']
relation_labels = ['applies to jurisdiction', 'field of work', 'native language', 'occupation', 'employer', 'owned by', 'instrument', 'located in the administrative territorial entity', 'genre', 'capital of', 'named after', 'religion', 'languages spoken, written or signed', 'headquarters location', 'country', 'manufacturer', 'developer', 'place of birth', 'twinned administrative body', 'place of death', 'record label', 'country of citizenship', 'location', 'subclass of', 'continent', 'instance of', 'capital', 'part of', 'original language of film or TV show', 'official language', 'position held', 'language of work or name', 'position played on team / speciality', 'original network', 'member of', 'shares border with', 'country of origin', 'has part', 'diplomatic relation', 'location of formation', 'work location']
fewshot_dir_path = 'case_based_10_train'
raw_dir_path = 'lama'

with open('data/type_file/bert.json', 'r') as f:
    relation_token = json.load(f)


def get_intype_rank(prediction, type_token_set):
    intype_rank = 0
    answer = prediction['obj_label']
    total = len(prediction)
    for i in range(total):
        word = prediction['topk'][i]['token']
        if word in type_token_set:
            intype_rank += 1
        if word == answer:
            break
    return intype_rank

total_inc_cnt, total_unchange_cnt, total_dec_cnt = 0, 0, 0
for relation, relation_label in zip(relations, relation_labels):
    with open(os.path.join(fewshot_dir_path, relation, relation + '_predictions.jsonl'), 'r') as f:
        fewshot_prediction_list = f.readlines()
        fewshot_prediction_list = [json.loads(pred) for pred in fewshot_prediction_list]
    with open(os.path.join(raw_dir_path, relation, relation + '_predictions.jsonl'), 'r') as f:
        raw_prediction_list = f.readlines()
        raw_prediction_list = [json.loads(pred) for pred in raw_prediction_list]
    type_token_set = set(relation_token[relation])
    inc_cnt, unchange_cnt, dec_cnt = 0, 0, 0
    for raw_prediction, fewshot_prediction in zip(raw_prediction_list, fewshot_prediction_list):
        raw_intype_rank = get_intype_rank(raw_prediction, type_token_set)
        fewshot_intype_rank = get_intype_rank(fewshot_prediction, type_token_set)
        if raw_intype_rank < fewshot_intype_rank:
            dec_cnt += 1
        elif raw_intype_rank > fewshot_intype_rank:
            inc_cnt += 1
        else:
            unchange_cnt += 1
    total = inc_cnt + unchange_cnt + dec_cnt
    inc_per, unchange_per, dec_per = inc_cnt * 100.0 / total, unchange_cnt * 100.0 / total, dec_cnt * 100.0 / total
    total_inc_cnt += inc_cnt
    total_unchange_cnt += unchange_cnt
    total_dec_cnt += dec_cnt
    print(
        relation,
        relation_label,
        inc_cnt,
        unchange_cnt,
        dec_cnt,
        inc_per,
        unchange_per,
        dec_per,
        sep='\t'
    )

total = total_inc_cnt + total_unchange_cnt + total_dec_cnt
mean_inc_per, mean_unchange_per, mean_dec_per = total_inc_cnt * 100.0 / total, total_unchange_cnt * 100.0 / total, total_dec_cnt * 100.0 / total
print(
    'mean',
    ' ',
    total_inc_cnt,
    total_unchange_cnt,
    total_dec_cnt,
    mean_inc_per,
    mean_unchange_per,
    mean_dec_per,
    sep='\t'
)