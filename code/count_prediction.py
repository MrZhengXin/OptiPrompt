import json
import collections

relations = ['P1001', 'P101', 'P103', 'P106', 'P108', 'P127', 'P1303', 'P131', 'P136', 'P1376', 'P138', 'P140', 'P1412', 'P159', 'P17', 'P176', 'P178', 'P19', 'P190', 'P20', 'P264', 'P27', 'P276', 'P279', 'P30', 'P31', 'P36', 'P361', 'P364', 'P37', 'P39', 'P407', 'P413', 'P449', 'P463', 'P47', 'P495', 'P527', 'P530', 'P740', 'P937']
dirs = ['wiki_uni', 'lama']

for dir_path in dirs:
    for r in relations:
        with open('%s/%s/%s_predictions.jsonl' % (dir_path, r, r), 'r') as f:
            results = f.readlines()
        pred_objs = []
        for res in results:
            pred_obj = json.loads(res)["topk"][0]["token"]
            pred_objs.append(pred_obj)
        counter = collections.Counter(pred_objs)
        with open('%s/%s/%s' % (dir_path, dir_path, r), 'w') as f:
            json.dump(dict(counter), fp=f)