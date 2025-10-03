from openai import OpenAI
import json
from utils.utils import deal_judge_new, write_jsonl, read_json
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    base_url="http://localhost:8000/v1",
    api_key = "114514"
)

def answer_match(res1, res2):

    model = "/home/gomall/models/Qwen2.5-3B-Instruct"
    prompt = 'Are the following two responses semantically equivalent? You should only output "Yes" or "No".\nResponse1:{response1}\nResponse2:{response2}\nYour judgement:'
    completion = client.chat.completions.create(
                    temperature=0,
                    model=model,
                    top_p=1,
                    messages=[
                        {'role': 'system', 'content': 'Your are a helpful assistant.'},
                        {'role': 'user', 'content': prompt.format(response1 = res1, response2 = res2)}
                    ], 
                    modalities=["text"],
                    stream=True,
                    stream_options={"include_usage": True}
                )
    res = ""
    for chunk in completion:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                res += content
    return 1-deal_judge_new(res)


consistency_input_path = "/home/gomall/work/MLLM/multi_answer_res/qwen2.5-7B/year_200/award_200/award_6a_200_res.jsonl"
rawanswer_input_path = "/home/gomall/work/MLLM/multi_answer_res/qwen2.5-7B/year_200/greedy/award_200/award_6a_200_greedy.jsonl"
output_path = "/home/gomall/work/MLLM/multi_answer_res/qwen2.5-7B/year_200/award_200/award_6a_200_consist.jsonl"
out = []
consistency_res = read_json(consistency_input_path)
consistency_res = [i for i in consistency_res if 'info' not in i]
consistency_res = [consistency_res[i]['Res'] for i in range(len(consistency_res))]
raw_answer = read_json(rawanswer_input_path)
raw_answer = [i for i in raw_answer if 'info' not in i]
for i in tqdm(range(len(consistency_res))):
    raw  = raw_answer[i]['Res']
    k = 0
    consistency_list = []
    for answer in consistency_res[i]:
        match = answer_match(answer[0], raw)
        k += match
        consistency_list.append(k)
    out.append({'Res': raw, 'consistency': consistency_list, 'reference':raw_answer[i]["reference"], "has_answer": raw_answer[i]['has_answer'], "question":raw_answer[i]["question"]})
    write_jsonl(out, output_path)





