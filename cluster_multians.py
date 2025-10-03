from openai import OpenAI
import json
from utils.utils import deal_judge_new, write_jsonl, read_json, has_answer
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    base_url="http://localhost:8000/v1",
    api_key = "114514"
)

def cluster_answers(answer_list):
    clusters = []
    
    for item in answer_list:
        answer = item
        matched = False
        
        # 检查当前答案是否与已有聚类中的任一答案匹配
        for cluster in clusters:
            # 只需要与聚类中的第一个答案比较即可
            if answer_match(answer, cluster[0]):
                cluster.append(item)
                matched = True
                break
                
        # 如果没有匹配的聚类，则创建新聚类
        if not matched:
            clusters.append([item])
    
    # 提取纯答案的聚类结果（去掉字典结构）
    result = []
    for cluster in clusters:
        result.append([item for item in cluster])
    
    return result


def answer_match(res1, res2):

    model = "/home/gomall/models/Qwen2.5-3B-Instruct"
    prompt = 'Are the following two responses semantically equivalent? You should only output "Yes" or "No".\nResponse1:{response1}\nResponse2:{response2}\nYour judgement:'
    msg = prompt.format(response1 = res1, response2 = res2)
    msg = msg[:1000] if len(msg) > 1000 else msg
    completion = client.chat.completions.create(
                    temperature=0,
                    model=model,
                    top_p=1,
                    messages=[
                        {'role': 'system', 'content': 'Your are a helpful assistant.'},
                        {'role': 'user', 'content': msg}
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

# 示例使用
if __name__ == "__main__":
    
    input_file = "multi_answer_res/mistral-7B/q1_8a/math_1ans.jsonl"
    output_file = "multi_answer_res/mistral-7B/q1_8a/consistency_res/math_1ans.jsonl"
    data = read_json(input_file)
    output_data = []
    for line in tqdm(data):
        res = line["Res"]
        answers = [str(item[0]) for item in res]
        clustered_res = cluster_answers(answers)
        line["cluster"] = clustered_res
        line["consistency"] = max([len(item) for item in clustered_res])
        acc = sum([has_answer([str(i) for i in line["reference"]], answer) for answer in answers]) / len(answers)
        line["acc"] = acc
        output_data.append(line)
    write_jsonl(output_data, output_file)
        