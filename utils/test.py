import json
path = "./data/mllm_datasets/visual7w/v7w.jsonl"
out = "./data/mllm_datasets/Dyn-QA/vqa_500_relative_localimg.jsonl"
f = open(path, 'r', encoding='utf-8')
lines = f.readlines()
raw = []
for line in lines:
    raw.append(json.loads(line))
new = []
for line in raw:
    if "info" not in line:
        line['image_url'] = line['image_url'][94:]
        new.append(line)

lines = []
for data in new:
    lines.append(json.dumps(data) + "\n")
with open(out, 'w', encoding='utf-8') as f:
    f.writelines(lines)
