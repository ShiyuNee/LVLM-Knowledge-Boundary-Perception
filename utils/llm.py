import time
import os
from .utils import deal_answer, deal_judge, deal_post, str2paras, deal_judge_new, has_answer, encode_image, write_json_from_start, encode_image_add_noise, output_completion, output_logprobs
from utils.prompt import get_evaluate_output_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
from openai import OpenAI
import copy
all_choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
false_terms=['false', 'wrong', 'no answer','not in']  # judging evaluation model's output.
true_terms=['true', 'correct', 'has answer']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    base_url="http://localhost:8000/v1",
    api_key = "114514"
)

# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key="sk-699e499c2fdb4bd39468b5753c7de854",
#     base_url="https://api.deepseek.com",
# )

multi_step_template = {
    "verb_2s_vanilla":'If you are sure your previous answer was accurate and correct, please say "certain", If you are not confident with the answer, please say "uncertain". Give ONLY "certain" or "uncertain", no other words or explanation.',
    "verb_2s_vanilla_re":'If you are not confident with the answer, please say "uncertain", If you are sure your previous answer was accurate and correct, please say "certain". Give ONLY "certain" or "uncertain", no other words or explanation.',
    "verb_2s_cot": 'If you are sure your previous answer was accurate and correct, please say "certain", If you are not confident with the answer, please say "uncertain". Analyse step by step, then provide your judgement.',
    "verb_2s_cot_re": 'If you are not confident with the answer, please say "uncertain", If you are sure your previous answer was accurate and correct, please say "certain". Analyse step by step, then provide your judgement.',
    "verb_2s_idt": "I don't think your answer is right, if you still think your answer is right, please say \"ceratin\". Otherwise, say \"uncertain\".",
    "verb_2s_pun": 'If you are sure your previous answer was accurate and correct, please say "certain", If you are not confident with the answer, please say "uncertain". You will be punished if the answer is not right but you say "certain".',
    "verb_2s_unsure": 'I\'m doubt the reliability of your answer. If you are sure your previous answer was accurate and correct, please say "certain", If you are not confident with the answer, please say "uncertain".',
    "prob_2s_vanilla":'Provide the probability that your answer is correct (0.0 to 1.0). Give ONLY the probability, no other words or explanation. For example:\n\nProbability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>',
    "prob_2s_pun":'Provide the probability that your answer is correct (0.0 to 1.0), you will be punished if the probability is high but the guess is wrong. Give ONLY the probability, no other words or explanation. For example:\n\nProbability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>',
    "prob_2s_idt":"I don't think your answer is right, provide the probability that your answer is correct (0.0 to 1.0). Give ONLY the probability, no other words or explanation. For example:\n\nProbability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>",
    "prob_2s_cot":'Provide the probability that your answer is correct (0.0 to 1.0). Before giving your answer, provide a step-by-step analyzation of your thought process. Then on a new line give the probability with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>',
}

class ApiGenerator:
    def __init__(self, args):
        self.args = args
        self.model = args.model_name
        self.batch_size = args.batch_size
        self.answer_match_model = args.answer_match_model
        self.writed_line = self.args.start_line
        self.ok_line = self.args.start_line
        self.cnt = 0
        self.problist = []
        if args.answer_judge == "NLI":
            self.NLItokenizer = AutoTokenizer.from_pretrained(self.answer_match_model)
            self.NLImodel = AutoModelForSequenceClassification.from_pretrained(self.answer_match_model).to(device)
    
    def calculate_res_mid(self):
        """
        保存输出结果
        """
        all_data = self.data.data # 所有数据, 需要算结果的数据可能是其中一部分
        res = []
        acc = 0
        begin = self.writed_line-self.args.start_line
        ##############################parameters of hallucination classify###########################
        image_err = 0
        lang_err = 0
        both_err = 0
        ##############################end parameters of hallucination classify###########################
        print(f'len of all data: {len(all_data)}')
        last_ok_line = self.ok_line
        start_idx = self.ok_line-self.args.start_line
        for idx in range(start_idx,len(all_data)):
            if idx not in self.data.idxs: # 不需要统计的数据
                res.append(all_data[idx])
            else:
                res_sample = {}
                if begin >= len(self.outputs):
                    break  # 防止越界
                if 'qa' in self.args.type:
                    res_sample['qa_prompt'] = self.data[begin]
                    res_sample['question'] = all_data[idx]['question']
                    if self.args.logprobs:
                        res_sample['logprobs'] = self.problist[begin]
                    if self.args.multi_step_type is not None:
                        res_sample['Res'] = self.outputs[begin]['Res'][0]
                        res_sample['judging'] = self.outputs[begin]['Res'][1]
                    else:
                        res_sample['Res'] = self.outputs[begin]['Res']
                    res_sample['reference'] = all_data[idx]['reference']
                    references = [str(data) for data in all_data[idx]['reference']]
                    if not self.args.using_consistency:
                        if self.args.answer_judge == "llm":
                            res_sample['has_answer'], judging_res = self.model_match_answer(res_sample['question'],
                                                                                            references,
                                                                                            res_sample['Res'])
                            if res_sample['has_answer'] is None:
                                print('don not validly judged!:')
                                print(judging_res)
                        elif self.args.answer_judge == "in_answer":
                            res_sample['has_answer'] = has_answer(references, res_sample['Res'])
                        elif self.args.answer_judge == "NLI":
                            res_sample['has_answer'] = self.model_match_answer_NLI(references,
                                                                                   res_sample['Res'],
                                                                                   threshold=0.35)
                        if 'prior' in self.args.type or 'post' in self.args.type: # verbalized confidence
                            res_sample['has_answer'] = deal_judge_new(res_sample['Res']) if 'mc' not in self.args.type else deal_judge_new(res_sample['Full_res'])
                        acc += res_sample['has_answer']
                elif self.args.type == 'judging_hal':
                    res_sample['question'] = all_data[idx]['question']
                    res_sample['reference'] = all_data[idx]['reference']
                    res_sample['Res'] = all_data[idx]['Res']
                    res_sample['judging_res'] = self.outputs[begin]['Res']
                    judging_res = res_sample['judging_res']
                    if "image" in judging_res.lower():
                        image_err += 1
                    elif "language" in judging_res.lower():
                        lang_err += 1
                    elif "both" in judging_res.lower():
                        both_err += 1
                elif self.args.type == 'q_gen' or 'image_disc' in self.args.type or self.args.type == 'choice_gen':
                    res_sample['question'] = all_data[idx]['question']
                    res_sample['Res'] = self.outputs[begin]['Res']
                res.append(res_sample)
                begin += 1
                self.writed_line += 1
            self.ok_line+=1
        if self.args.type == 'judging_hal':
            sum_cnt = image_err + lang_err + both_err
            print(f'image err rate: {image_err/sum_cnt}')
            print(f'language err rate: {lang_err/sum_cnt}')
            print(f'both err rate: {both_err/sum_cnt}')
        print(f'ready to save line: {self.ok_line}')
        write_json_from_start(self.args.outfile, last_ok_line, res)
        print("saved!")

    def calculate_res(self):
        """
        保存输出结果
        """
        all_data = self.data.data # 所有数据, 需要算结果的数据可能是其中一部分
        res = []
        begin = 0
        acc = 0
        ##############################parameters of hallucination classify###########################
        image_err = 0
        lang_err = 0
        both_err = 0
        ##############################end parameters of hallucination classify###########################
        print(f'len of all data: {len(all_data)}')
        for idx in range(len(all_data)):
            if idx not in self.data.idxs: # 不需要统计的数据
                res.append(all_data[idx])
            else:
                res_sample = {}
                if begin >= len(self.outputs):
                    break  # 防止越界
                if 'qa' in self.args.type:
                    res_sample['qa_prompt'] = self.data[begin]
                    res_sample['question'] = all_data[idx]['question']
                    res_sample['Res'] = self.outputs[begin]['Res']
                    if self.args.answer_match_model:
                        res_sample['has_answer'], judging_res = self.model_match_answer(res_sample['question'],
                                                                                        all_data[idx]['reference'],
                                                                                        res_sample['Res'])
                        if res_sample['has_answer'] is None:
                            print('don not validly judged!:')
                            print(judging_res)
                    else:
                        res_sample['has_answer'] = has_answer(all_data[idx]['reference'], res_sample['Res'])
                    res_sample['reference'] = all_data[idx]['reference']
                    if 'prior' in self.args.type or 'post' in self.args.type: # verbalized confidence
                        res_sample['has_answer'] = deal_judge_new(res_sample['Res']) if 'mc' not in self.args.type else deal_judge_new(res_sample['Full_res'])
                    acc += res_sample['has_answer']
                elif self.args.type == 'judging_hal':
                    res_sample['question'] = all_data[idx]['question']
                    res_sample['reference'] = all_data[idx]['reference']
                    res_sample['Res'] = all_data[idx]['Res']
                    res_sample['judging_res'] = self.outputs[begin]['Res']
                    judging_res = res_sample['judging_res']
                    if "image" in judging_res.lower():
                        image_err += 1
                    elif "lm" in judging_res.lower():
                        lang_err += 1
                    elif "both" in judging_res.lower():
                        both_err += 1
                
                res.append(res_sample)
                begin += 1
        if self.args.type == 'judging_hal':
            sum_cnt = image_err + lang_err + both_err
            print(f'image err rate: {image_err/sum_cnt}')
            print(f'language err rate: {lang_err/sum_cnt}')
            print(f'both err rate: {both_err/sum_cnt}')
        print(f'processed data count: {begin}')
        print(f'accuracy: {acc / begin}')
        return res, acc / begin

            

    def load_data(self, data):
        self.data = data
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=self.batch_size)

    def model_classify_hal_batch(self, inputs):
        responses = []
        for input_data in inputs:
            if self.args.local_image:  # the images are in your computer
                base64_image = encode_image(input_data["image_url"])
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text":  'Given a question, a image,list of ground truth answers and a MLLM wrong statement\
                        about the question and image pairs. Based on the question image pairs with the ground truth list, \
                        you should determine WHY the statement is wrong. Your judgement should based on the following entries.\n\
                        If you think the error is caused by the following three reasons, output"Image":\n\
                        1.The model failed to recognize the objects or details in the picture, resulting in no corresponding answers in the response or refuse to answer.\n\
                        Example:\n\
                        Question: What does the green vehicle have?\nGround truth answer: Green writing on the door.\nImage about:A Green vehicle with \'Green\' writing on the door.\nMLLM Statement: The image does not contain a green vehicle, the vehicle is white. \nYour Judgement:\'Image\'\n\
                        2.The model either misinterpreted or failed to capture the relationships between the elements in the image.\n\
                        Example:\n\
                        Question: What are the boys doing?\nGround truth answer: They are playing.\nImage about:Two boys playing with each other.\nMLLM Statement: They are fighting. \nYour Judgement:\'Image\'\n\
                        3.The model has problems in identifying the attributes (such as color, volume, etc.) and states (such as position, etc.) of the elements in the picture.\n​\
                        Example:\n\
                        Question: How many bottles on the desk?\nGround truth answer: Three.\nImage about:Three bottles on a desk and two bottles under it.\n MLLM Statement: Two bottles. \nYour Judgement:\'Image\'\n\
                        else if you think the error is caused by the following three reasons, output"LM":\n\
                        1.There are problems with the logical thinking chain of the model in generating answers.\n\
                        Example:\n\
                        Question: what is the result of the math problem in the Image?\nGround truth answer: 9.\nImage about:A black board with \'5*5-4*4\' on it.\n MLLM Statement: First calculate multiply:5*5=25,4*4=16,then substract, the answer is 8.\nYour Judgement:\'LM\'\n\
                        2.Model doesn\'t contain the right knowledge about the question,induce knowledge error in the answer.​\n\
                        Example:\n\
                        Question: When was he born?\nGround truth answer: 1879.\nImage about:Albert Einstein.\n MLLM Statement: The man in the image is Einstein, he born in 1900. \nYour Judgement:\'LM\'\n\
                        3.The model fails to correctly understand the question.​\n\
                        Example:\n\
                        Question: How many kites in the sky?\nGround truth answer: 20.\nImage about:20 kites flying in the sky.\n MLLM Statement: The image shows many kites flying in the blue sky. \nYour Judgement:\'LM\'\n\
                        Remember, only output "Image" or "LM" without any other words.'}]},
                        {"role": "user", "content": [
                            # PNG图像：  f"data:image/png;base64,{base64_image}"
                            # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                            # WEBP图像： f"data:image/webp;base64,{base64_image}"
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": input_data["query"]}
                        ]}
                    ],
                    modalities=["text"],
                    stream=True,
                    # stream_options={"include_usage": True}
                )
            else:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text":   'Given a question, a image,list of ground truth answers and a MLLM wrong statement\
                        about the question and image pairs. Based on the question image pairs with the ground truth list, \
                        you should determine WHY the statement is wrong. Your judgement should based on the following entries.\n\
                        If you think the error is caused by the following three reasons, output"Image":\n\
                        1.The model failed to recognize the objects or details in the picture, resulting in no corresponding answers in the response or refuse to answer.\n\
                        2.The model either misinterpreted or failed to capture the relationships between the elements in the image.\n\
                        3.The model has problems in identifying the attributes (such as color, volume, etc.) and states (such as position, etc.) of the elements in the picture.\n​\
                        else if you think the error is caused by the following three reasons, output"LM":\n\
                        1.There are problems with the logical thinking chain of the model in generating answers.\n\
                        2.Model doesn\'t contain the right knowledge about the question,induce knowledge error in the answer.\n\
                        3.The model fails to correctly understand the question.​\n\
                        Remember, only output "Image" or "LM" without any other words.'}]},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": input_data["image_url"]}},
                            {"type": "text", "text": input_data["query"]}
                        ]}
                    ],
                    modalities=["text"],
                    stream=True,
                    # stream_options={"include_usage": True}
                )
            # 收集响应
            response = ""
            for chunk in completion:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        response += content
            responses.append(response)
        return responses

    def model_match_answer(self, question, answers, state):
        content = get_evaluate_output_prompt(question, answers, state, self.args)
        completion = client.chat.completions.create(
                model=self.answer_match_model,
                messages=[
                    {'role': 'system', 'content': 'Given a question and a list of its ground truth answer,\
                    you should output True if only at least one of the answers in the ground truth answer list\
                    is consistent with the statement. Otherwise output False. Remember, only output True or False without\
                    any other words.'},
                    {'role': 'user', 'content': content}
                ], 
                modalities=["text"],
                stream=True,
                # stream_options={"include_usage": True}
        )
        res = ""
        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    res += content
        for term in true_terms:
            if term.lower() in res.lower():
                return True, res
        for term in false_terms:
            if term.lower() in res.lower():
                return False, res
        return None, res
  
    def model_match_answer_NLI(self, answers, res, threshold):
        for answer in answers:
            inputs = self.NLItokenizer(res, str(answer), return_tensors="pt", truncation=True).to(device)
            # 推理
            outputs = self.NLImodel(**inputs)
            logits = outputs.logits
            # 计算softmax以获得每个类别的概率
            entailment_prob = torch.softmax(logits, dim=1)[0][2].item()
            if entailment_prob >= threshold:
                return True
        return False
    
    def get_res(self):
        self.outputs = []
        for batch in tqdm(self.dataloader):
            if self.args.model_type == 'mllm':
                queries, image_urls = batch['query'], batch['image_url']
                if self.args.multi_step_type:
                    res = batch['Res']
                else:
                    res = [None for i in range(len(queries))]
                if not self.args.consistency_perturb:
                    inputs = [{"query":queries[i] , "image_url":image_urls[i], "Res":res[i] } for i in range(len(queries))]
                if self.args.type == "judging_hal":
                    outs = self.model_classify_hal_batch(inputs)
                elif self.args.using_consistency:
                    if self.args.consistency_perturb:
                        outs = self.process_batch_mllm_consistency(batch, 1.0, 1.0)
                    else:
                        outs = self.process_batch_mllm_consistency(inputs, 1.0, 1.0)
                else:
                    outs = self.process_batch_mllm(inputs, 0, 1.0)
            elif self.args.model_type == 'llm':
                if self.args.using_consistency:
                    outs = self.process_batch_llm_consistency(batch, 1.0, 1.0)
                else:
                    outs = self.process_batch_llm(batch, 0, 1.0)
            self.process_res(outs)
            self.calculate_res_mid()

        print(f'len of outputs: {len(self.outputs)}')
        return 0,0
        # return self.calculate_res()
    def process_batch_llm(self, batch, temperature, top_p):
        responses = []
        idx = 0
        if self.args.multi_step_type is None:
            new_batch = batch
        else:
            new_batch = batch['question']
        for input_data in new_batch:
            if self.args.multi_step_type is None:
                messages = [
                        {'role': 'system', 'content': 'Your are a helpful assistant.'},
                        {'role': 'user', 'content': input_data}
                    ]
            else:
                messages = [
                        {'role': 'system', 'content': 'Your are a helpful assistant.'},
                        {'role': 'user', 'content': batch['question'][idx]},
                        {"role": "assistant", "content": batch['Res'][idx]},
                        {"role": "user", "content": multi_step_template[self.args.multi_step_type]}
                    ]
                
            completion = client.chat.completions.create(
                temperature=temperature,
                model=self.model,
                top_p=top_p,
                messages=messages, 
                modalities=["text"],
                stream=self.args.stream_output,
                # stream_options={"include_usage": True},
                logprobs=True,
                **({"seed": 114514} if not self.args.using_consistency else {})
            )
            # print(self.args.logprobs)
            res, probs = output_completion(completion, self.args.stream_output, self.args.logprobs)
            res = (batch['Res'][idx], res) if self.args.multi_step_type is not None else res
            self.problist.append(probs)
            responses.append(res)
            idx += 1
        return responses
            
    def process_batch_llm_consistency(self, batch, temperature, top_p):
        res = []
        if self.args.consistency_perturb:
            processed_batch = [[] for _ in range(len(batch[0]))]
            for i in range(len(batch[0])):
                for j in range(len(batch)):
                    processed_batch[i].append(batch[j][i])
            batch_res=[]
            for questions in processed_batch:
                responses=[]
                for i in range(self.args.consistency_num):
                    completion = client.chat.completions.create(
                        temperature=temperature,
                        model=self.model,
                        top_p=top_p,
                        messages=[
                            {'role': 'system', 'content': 'Your are a helpful assistant.'},
                            {'role': 'user', 'content': questions[i]}
                        ], 
                        modalities=["text"],
                        stream=self.args.stream_output,
                        # stream_options={"include_usage": True},
                        **({"seed": 114514} if not self.args.using_consistency else {})
                    )
                    res = output_completion(completion, self.args.stream_output, self.args.logprobs)
                    responses.append(res)
                batch_res.append(responses)
            return batch_res
        else:
            batch_res = []
            for input_data in batch:
                responses = []
                for i in range(self.args.consistency_num):
                    completion = client.chat.completions.create(
                        temperature=temperature,
                        model=self.model,
                        top_p=top_p,
                        messages=[
                            {'role': 'system', 'content': 'Your are a helpful assistant.'},
                            {'role': 'user', 'content': input_data}
                        ], 
                        modalities=["text"],
                        stream=self.args.stream_output,
                        # stream_options={"include_usage": True},
                        **({"seed": 114514} if not self.args.using_consistency else {})
                    )
                    res = output_completion(completion, self.args.stream_output, self.args.logprobs)
                    responses.append(res)
                batch_res.append(responses)
            return batch_res

    def process_batch_mllm_consistency(self, batch, temperature, top_p):
        res = []
        if self.args.consistency_perturb:
            inputs = batch['query']
            images = batch['image_url']
            processed_batch = [[] for _ in range(len(inputs[0]))]
            for i in range(len(inputs[0])):
                for j in range(len(inputs)):
                    processed_batch[i].append(inputs[j][i])
            batch_res=[]
            j = 0
            for questions in processed_batch:
                responses=[]
                if self.args.image_noise:
                    start_noise = self.args.image_noise_start
                    step_noise = self.args.image_noise_step
                    for i in range(self.args.consistency_num):
                        res = self.mllm_completion_create({'query':questions[i],'image_url': images[j]}, temperature, top_p,start_noise)
                        responses.append(res)
                        start_noise += step_noise
                else:
                    for i in range(self.args.consistency_num):
                        res = self.mllm_completion_create({'query':questions[i],'image_url': images[j]}, temperature, top_p)
                        responses.append(res)
                batch_res.append(responses)
                j += 1
            return batch_res
        else:
            batch_res = []
            for input_data in batch:
                responses = []
                if self.args.image_noise:
                    start_noise = self.args.image_noise_start
                    step_noise = self.args.image_noise_step
                    for i in range(self.args.consistency_num):
                        res = self.mllm_completion_create(input_data, temperature, top_p,start_noise)
                        responses.append(res)
                        start_noise += step_noise
                else:
                    for i in range(self.args.consistency_num):
                        res = self.mllm_completion_create(input_data, temperature, top_p)
                        responses.append(res)
                batch_res.append(responses)
            return batch_res

    def process_batch_mllm(self, inputs, temperature, top_p):
        responses = []
        for input_data in inputs:
            response=self.mllm_completion_create(input_data, temperature, top_p)
            responses.append(response)
        return responses
    def process_res(self, outs):
        for res in outs:
            self.outputs.append({'Res':res})

    def mllm_completion_create(self, input_data, temperature, top_p,noise_level=None):
        try:
            
            if self.args.local_image:  # the images are in your computer
                if noise_level is not None:
                    base64_image = encode_image_add_noise(input_data["image_url"], noise_level)
                else:
                    base64_image = encode_image(input_data["image_url"])
                if "deepseek-vl2-tiny" in self.args.model_name:
                     messages = [
                                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                                {"role": "user", "content": [
                                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                    {"type": "image_url", "image_url": {"url":input_data["image_url"]}},
                                    {"type": "text", "text": input_data["query"]}
                                ]}
                            ]
                else:
                    messages = [
                                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                                {"role": "user", "content": [
                                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                    {"type": "text", "text": input_data["query"]}
                                ]}
                            ]
                if self.args.multi_step_type is None:
                    completion = client.chat.completions.create(
                        temperature=temperature,
                        model=self.model,
                        top_p=top_p,
                        max_tokens = 200,
                        messages=messages,
                        modalities=["text"],
                        stream=self.args.stream_output,
                        # stream_options={"include_usage": True},
                        logprobs=True,
                        **({"seed": 114514} if not self.args.using_consistency else {}),
                    )
                
            else:
                messages = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": input_data["image_url"]}},
                            {"type": "text", "text": input_data["query"]}
                        ]}
                    ]
                if self.args.multi_step_type is None:
                    completion = client.chat.completions.create(
                        temperature=temperature,
                        top_p=top_p,
                        model=self.model,
                        messages=messages,
                        modalities=["text"],
                        stream=self.args.stream_output,
                        max_tokens = 200,
                        # stream_options={"include_usage": True},
                        logprobs=True,
                        **({"seed": 114514} if not self.args.using_consistency else {}),
                    )
            # 收集响应
            if self.args.multi_step_type is not None:
                messages.append({"role": "assistant", "content": input_data['Res']})
                messages.append({"role": "user", "content": multi_step_template[self.args.multi_step_type]})

                completion = client.chat.completions.create(
                    temperature=temperature,
                    model=self.model,
                    top_p=top_p,
                    messages=messages,
                    modalities=["text"],
                    max_tokens = 200,
                    stream=self.args.stream_output,
                    # stream_options={"include_usage": True},
                    **({"seed": 114514} if not self.args.using_consistency else {})
                    
                    )
            
        except Exception as e:
            print('\nError\t', e, "url",input_data['image_url'], '\tReturn')
            return    
        response, probs = output_completion(completion, self.args.stream_output, self.args.logprobs)
        self.problist.append(probs)
        if self.args.multi_step_type:
            return (input_data["Res"], response)
        return response




class Generater:
    def __init__(self, args):
        self.args = args
        if '7b' or '8b' in self.args.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(args.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.model_path).half()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.outputs = []
        self.eos_id_dict = {
            'llama2-7b-chat': self.tokenizer.eos_token_id,
            'llama3-8b-instruct': self.tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0],
            'qwen2-7b-instruct': self.tokenizer.eos_token_id,
            'llama2-13b-chat': self.tokenizer.eos_token_id,
        }
        print('load generater finish.')

    def load_data(self, data):
        self.data = data
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=self.batch_size)
        if self.args.task == 'mmlu' or self.args.task == 'tq':
            self.choice_cnt = self.data.choice_cnt
        
    
    def get_res(self):
        self.outputs = []
        device = torch.device('cuda')
        self.device = device
        self.model.to(device)
        for batch in tqdm(self.dataloader):
            batch = self.tokenizer(batch, return_tensors='pt', padding=True).to(device)
            input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
            outs = self.model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=self.args.max_new_tokens, 
                                       output_attentions=self.args.attn_weights, return_dict_in_generate=True, output_scores=True, output_hidden_states=self.args.hidden_states, 
                                       pad_token_id=0, top_p=1.0, temperature=1, do_sample=False)
            if self.args.task == 'mmlu' or self.args.task == 'tq':
                self.process_res_multi_choice(outs, input_ids) # 得到一个batch的结果
            else:
                self.process_res(outs, input_ids)
        print(f'len of outputs: {len(self.outputs)}')
        return self.calculate_res()
    
    def process_res(self, outs, inputs):
        """
        按batch处理模型generate输出, 得到输出文本,每个token的概率,以及每个token的entropy
        Input:
            - outs: generate输出结果
            - inputs: generate的input_ids
        Return:
            - 输出列表, 每个元素是一个字典
            {
                'Res': 生成文本,
                'Log_p':{
                    'tokens':生成的每个token,
                    'token_probs': 生成的每个token的概率,
                    'token_entropy': 生成每个token时对应的vocab空间的entropy
                }
            }
        """
        # attention和scores都不包含输入
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        input_len = inputs.shape[-1]
        bt_size = inputs.shape[0]
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        # print(f'text: {self.tokenizer.batch_decode(new_ids, skip_sepcial_tokens=True)}')
        end_idx = self.get_generation_end(new_ids)
        print(f'end_idx: {end_idx}')
        # for idx in range(len(new_ids)):
        #     print(self.tokenizer.convert_ids_to_tokens(new_ids[idx][:end_idx[idx]]))
        # 存储概率最大的token_id, 存储对应的probs, 存储seqs对应probs. 当且仅当使用greedy search时, top_indices=outs['sequence']
        top_indices, top_scores, ans_scores, ans_entropy = self.get_generated_tokens_probs_entropy(scores, new_ids, bt_size)

        if self.args.hidden_states:
            hidden_modes = self.args.hidden_idx_mode.split(',')
            all_modes_hidden_state = [{} for _ in range(bt_size)]
            for mode in hidden_modes:
                if mode == 'ans': #不支持提取answer部分第一个token的hidden state
                    raise ValueError('Do not support hidden_mode=ans for free-form qa')
                if mode == 'every': # 得到ans token在每一层的概率, 每一层的top-1 token
                    probs_for_generated_tokens, tokens_for_each_layer = self.get_token_and_prob_for_each_pos(outs, bt_size, end_idx) #(bt_size, layers, ans_len)
                else:
                    if mode == 'conf':
                        pos_idx = self.get_confidence_idx(outs, inputs, end_idx)
                    else:
                        pos_idx = self.get_need_idx_for_generation(top_scores, end_idx, mode)
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, pos_idx, mode)
                    for bt in range(bt_size):
                        all_modes_hidden_state[bt][mode] = hidden_states[bt]
                

        for bt in range(bt_size):
            # print(f'ans: {self.tokenizer.decode(new_ids[bt][:end_idx[bt]])}')
            temp_res = ({
                'Res': self.tokenizer.decode(new_ids[bt][:end_idx[bt]]).strip(),
                'Log_p':{
                    'tokens': new_ids[bt][:end_idx[bt]].tolist(),
                    'token_probs': ans_scores[bt][:end_idx[bt]].tolist(),
                    'token_entropy': ans_entropy[bt][:end_idx[bt]].tolist()
                }
            })
            if self.args.hidden_states:
                if self.args.hidden_idx_mode == 'every':
                    temp_res['probs_for_generated_tokens'] = probs_for_generated_tokens[bt]
                    temp_res['tokens_for_each_layer'] = tokens_for_each_layer[bt]
                else:
                    temp_res['hidden_states'] = all_modes_hidden_state[bt]

            self.outputs.append(temp_res)

    def process_res_multi_choice(self, outs, inputs):
        """
        对多选问题,得到在输出的token上的结果,概率,entropy,hidden_state等信息
        Input:
            - outs:
            - inputs:
        Return:
            - 
        """
        choices = all_choices[:self.choice_cnt] + all_choices[:self.choice_cnt]
        # choices = ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'] # token可能有A和(A, 长度为8是为了对应
        if self.args.model_name in ['llama3-8b-instruct', 'qwen2-7b-instruct']:
            choices = all_choices[:self.choice_cnt] + all_choices[:self.choice_cnt] + all_choices[:self.choice_cnt]
            # choices = ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E']
        input_len = inputs.shape[-1]
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        end_idx = self.get_generation_end(new_ids)
        # print(f'text: {self.tokenizer.batch_decode(new_ids, skip_sepcial_tokens=True)}')
        print(f'end idx: {end_idx}')
        # 找到choice出现位置,以及对应的token id
        ans_token_idx, choices_idx = self.get_choice_idx(outs, inputs, end_idx)
        print(f'answer idx: {ans_token_idx}')
        need_scores = []
        bt_size = inputs.shape[0]
        for bt in range(bt_size):
            need_scores.append(scores[ans_token_idx[bt]][bt]) # vocab_size
        need_scores = torch.stack(need_scores)
        probs = nn.Softmax(dim=-1)(need_scores) # 词表中所有token概率
        next_token_probs = probs[:, choices_idx] # batch_size, 8
        entropy = torch.sum(-(probs * torch.log2(probs)), dim=-1) # batch_size, 8
        max_scores, max_indices = torch.max(next_token_probs, dim=-1) # 生成token
        # 得到所有token对应的prob,为提取min-prob token对应hidden state作准备
        _, top_scores, _, _ = self.get_generated_tokens_probs_entropy(scores, new_ids, bt_size)

        if self.args.attn_weights: 
            attentions = self.get_attn_multi_choice(outs, bt_size, ans_token_idx)

        if self.args.hidden_states:
            # 若有多种mode需要记录,则一次性记录所有mode的hidden state
            hidden_modes = self.args.hidden_idx_mode.split(',')
            all_modes_hidden_state = [{} for _ in range(bt_size)]
            for mode in hidden_modes:
                if mode == 'every': # 得到ans token在每一层的概率, 每一层的top-1 token
                    raise ValueError('Do not need to specify hidden_idx_mode=every for multi-choice qa')
                elif mode == 'ans': # 取response中ans的first token
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, ans_token_idx, mode)
                else:
                    if mode == 'conf':
                        pos_idx = self.get_confidence_idx(outs, inputs, end_idx)
                    else:
                        pos_idx = self.get_need_idx_for_generation(top_scores, end_idx, mode)
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, pos_idx, mode)
                for bt in range(bt_size):
                    all_modes_hidden_state[bt][mode] = hidden_states[bt]
        
        for bt in range(bt_size):
            temp_res = {
                'Res': choices[max_indices[bt]],
                'Full_res': self.tokenizer.decode(new_ids[bt][:end_idx[bt]]).strip(),
                'Log_p':{
                    'token probs': next_token_probs[bt].tolist(),# choices prob
                    'token_entropy': float(entropy[bt]), # real entropy
                },
                'end_idx': end_idx[bt]
            }
            if self.args.hidden_states:
                temp_res['hidden_states'] = all_modes_hidden_state[bt]
            if self.args.output_states:
                temp_res['output_states'] = probs[bt]
            if self.args.attn_weights:
                temp_res['attn_weights'] = attentions[bt]
            self.outputs.append(temp_res)

    def calculate_res(self):
        """
        保存输出结果
        """
        all_data = self.data.data # 所有数据, 需要算结果的数据可能是其中一部分
        res = []
        begin = 0
        acc = 0
        print(f'len of all data: {len(all_data)}')
        for idx in range(len(all_data)):
            if idx not in self.data.idxs: # 不需要统计的数据
                res.append(all_data[idx])
            else:
                res_sample = {}
                if 'qa' in self.args.type:
                    res_sample['qa_prompt'] = self.data[begin]
                    res_sample['Res'] = self.outputs[begin]['Res']
                    res_sample['Log_p'] = self.outputs[begin]['Log_p']
                    if self.args.task == 'mmlu' or self.args.task == 'tq':
                        res_sample['question'] = self.data.format_example(all_data, idx, include_answer=False)
                        res_sample['has_answer'] = res_sample['Res'] == all_data[idx][-1]
                        res_sample['reference'] = all_data[idx][-1]
                        res_sample['end_idx'] = self.outputs[begin]['end_idx']
                        res_sample['Full_res'] = self.outputs[begin]['Full_res']
                    else:
                        res_sample['question'] = all_data[idx]['question']
                        res_sample['has_answer'] = has_answer(all_data[idx]['reference'], res_sample['Res'])
                        res_sample['reference'] = all_data[idx]['reference']
                    if 'prior' in self.args.type or 'post' in self.args.type: # verbalized confidence
                        res_sample['has_answer'] = deal_judge_new(res_sample['Res']) if 'mc' not in self.args.type else deal_judge_new(res_sample['Full_res'])
                    if self.args.attn_weights:
                        res_sample['attn_weights'] = self.outputs[begin]['attn_weights'].tolist()
                    if self.args.hidden_states:
                        if self.args.hidden_idx_mode == 'every':
                            res_sample['probs_for_generated_tokens'] = self.outputs[begin]['probs_for_generated_tokens']
                            res_sample['tokens_for_each_layer'] = self.outputs[begin]['tokens_for_each_layer']
                        else:
                            res_sample['hidden_states'] = self.outputs[begin]['hidden_states']
                    if self.args.output_states:
                        res_sample['output_states'] = self.outputs[begin]['output_states'].tolist()
                    acc += res_sample['has_answer']
                res.append(res_sample)
                begin += 1
        print(f'processed data count: {begin}')
        print(f'accuracy: {acc / begin}')
        return res, acc / begin
    
    def get_hidden_states_for_given_pos(self, outs, bt_size, need_idx, mode='first'):
        """
        得到指定位置token生成时每一层的hidden_state
        Input:
            - out: generate结果
            - bt_size: batch size
            - need_idx: 每个batch生成结果中,需要获取hidden state的位置
            - need_layers: 需要获取的hidden states所在的层
        Return:
            - res: 每一层对应的hidden states, (batch_size, layers, hidden_dim)
        Note:
            - outs['hidden_states'] tuples of (genetared_token, layer)->(bs, generated_len, hidden_dim)
        """
        if self.args.need_layers == 'last':
            need_layers = [-1]
        elif self.args.need_layers == 'all':
            need_layers = range(len(outs['hidden_states'][0]))
        elif self.args.need_layers == 'mid':
            need_layers = [int(len(outs['hidden_states'][0]) / 2)]
        else:
            raise ValueError('Specify the wrong need_layers')
        # print(need_layers)
        
        res = [[] for _ in range(bt_size)]
        for bt in range(bt_size): # 遍历sample
            temp_idx = need_idx[bt] # 当前sample需要考虑的token的idx
            # print(f'need layers: {need_layers}')
            if type(temp_idx) != list: # 只需要取一个token
                for layer in need_layers: # 该token的每一层
                    hidden_states = outs['hidden_states'][temp_idx][layer][bt][-1] # bs, generated_len(input_len or 1), hidden_size
                    res[bt].append(hidden_states.to(torch.float16).tolist())
            else: # 取所有token
                for layer in need_layers: # 该token的每一层
                    temp_res = []
                    for item in temp_idx: # 所有需要考虑的tokens
                        temp_res.append(outs['hidden_states'][item][layer][bt][-1])
                    temp_res = torch.stack(temp_res)
                    if mode == 'avg':
                        res[bt].append(torch.mean(temp_res, dim=0).to(torch.float16).tolist())
                    elif mode == 'dim_min': # hidden state不同维度取min
                        res[bt].append(torch.min(temp_res, dim=0)[0].to(torch.float16).tolist())
                    elif mode == 'dim_max':
                        res[bt].append(torch.max(temp_res, dim=0)[0].to(torch.float16).tolist())
        return res

    def get_attn_multi_choice(self, outs, bt_size, need_idx):
        """
        提取选项生成时的各层attention weights
        Input:
            - out: generate结果
            - bt_size: batch size
            - need_idx: 每个batch生成结果中,选项token的idx
        Return:
            - res: 每一层中所有attn_head的注意力权重, (batch_size, layers, num_head, context_len)
        Note:
            - outs['attentions'] tuples of (genetared_token, layer)->(bs, num_head, generated_len, context_len)
        """
        res = [[] for _ in range(bt_size)]
        for bt in range(bt_size):
            temp_idx = need_idx[bt]
            for layer in range(len(outs['attentions'][temp_idx])): # temp_idx处token对应的所有层
                attentions = outs['attentions'][temp_idx][layer][bt, :, -1] # bs, head_num, seq_len(input_len)
                res[bt].append(attentions.tolist())
        return res

    def get_choice_idx(self, outs, inputs, end_idx):
        """
        找到每个样本中choice出现的位置
        """
        batch_size, input_len = inputs.shape
        # llama3中, 'A'和' A'不是一个表示
        choices = all_choices[:self.choice_cnt] + ['(' + item + ')' for item in all_choices[:self.choice_cnt]]
        # choices = ['A', 'B', 'C', 'D', 'E', '(A)', '(B)', '(C)', '(D)', '(E)']
        if self.args.model_name in ['llama3-8b-instruct', 'qwen2-7b-instruct']:
            choices = all_choices[:self.choice_cnt] + ['(' + item + ')' for item in all_choices[:self.choice_cnt]] + [' ' + item for item in all_choices[:self.choice_cnt]]
            # choices = ['A', 'B', 'C', 'D', 'E', '(A)', '(B)', '(C)', '(D)', '(E)', ' A', ' B', ' C', ' D', ' E']
        out_idx = [0 for _ in range(batch_size)] # 没找到就默认为第一个token
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        new_token_ids = seqs[:, input_len:]

        choices_idx = self.tokenizer(choices)['input_ids']
        if self.args.model_name in ['llama2-7b-chat', 'llama2-13b-chat']:
            # ['<s>', '_A'],  ['<s>', '(', 'A', ')']
            choices_idx = [item[1] if len(item) == 2 else item[2] for item in choices_idx] # _A, A等的token_id
            #['_A'], ['(A', ')']
        elif self.args.model_name in ['llama3-8b-instruct', 'qwen2-7b-instruct']:
            choices_idx = [item[0] for item in choices_idx]
        for bt in range(batch_size): # 遍历batch
            for idx in range(end_idx[bt]): # 一个序列中token
                token_id = new_token_ids[bt][idx]
                if token_id in choices_idx: # 第一个出现选项的位置
                    out_idx[bt] = idx
                    break
        return out_idx, choices_idx      

    def get_need_idx_for_generation(self, probs, end_idx, mode):
        """
        根据mode找到需要探测的token的index
        Input:
            - mode: 
                - first, last, min, avg - 得到需要的token的idx
                - dim_min, dim_max - 得到所有token的idx, 后续在hidden_dim上取min/max
        """ 
        res_idx = []
        bt_size = probs.shape[0]
        text_len = probs.shape[1]
        assert mode in ['first', 'last', 'avg', 'min', 'dim_min', 'dim_max']
        if mode == 'first':
            res_idx = torch.zeros(bt_size, dtype=torch.int) # 全选第一个位置
        elif mode == 'last':
            res_idx = [item if item != text_len else item - 1 for item in end_idx] # 全选最后一个位置
        elif mode == 'min':
            temp_idx = [item + 1 if item != text_len else item for item in end_idx]
            for bt in range(bt_size):
                min_prob, min_index = torch.min(probs[bt][:temp_idx[bt]], dim=-1) # batch_size
                res_idx.append(min_index)
        elif mode == 'avg' or mode == 'dim_min' or mode == 'dim_max':
            for bt in range(bt_size):
                if end_idx[bt] == text_len:
                    res_idx.append(list(range(end_idx[bt])))
                else:
                    res_idx.append(list(range(end_idx[bt] + 1)))
        return res_idx
    
    def get_token_and_prob_for_each_pos(self, outs, bt_size, end_idx):
        """
        得到每个位置每一层top-1 token(early exit), 最终生成的token在每一层的概率
        """
        probs_for_generated_token = [[] for _ in range(bt_size)] # 最终生成的token在每一层对应的概率
        tokens_for_each_pos = [[] for _ in range(bt_size)] #
        for bt in range(bt_size):
            end_pos = end_idx[bt]
            for pos in range(end_pos):
                hidden_states_for_all_layers = []
                for layer in range(len(outs['hidden_states'][pos]))[1:]:
                    hidden_states = outs['hidden_states'][pos][layer][bt][-1] # hidden_size
                    hidden_states_for_all_layers.append(hidden_states)
                hidden_states_for_all_layers = torch.stack(hidden_states_for_all_layers) # (layers, hidden_dim)
                probs = nn.Softmax(dim=-1)(self.model.lm_head(hidden_states_for_all_layers))
                max_value_for_each_layer, max_token_for_each_layer = torch.max(probs, dim=-1)
                tokens_for_each_pos[bt].append(self.tokenizer.convert_ids_to_tokens(max_token_for_each_layer))
                generated_token = max_token_for_each_layer[-1]
                probs_for_generated_token[bt].append(probs[:, generated_token])
            
            probs_for_generated_token[bt] = torch.stack(probs_for_generated_token[bt]).t().tolist()
            probs_for_generated_token[bt] = [[round(element, 4) for element in row] for row in probs_for_generated_token[bt]]
            tokens_for_each_pos[bt] = [[tokens_for_each_pos[bt][j][i] for j in range(len(tokens_for_each_pos[bt]))] for i in range(len(tokens_for_each_pos[bt][0]))]
        return probs_for_generated_token, tokens_for_each_pos
    
    def get_generation_end(self, generated_tokens):
        # generated_tokens batch_size, new_seq_len
        text_len = generated_tokens.shape[-1]
        end_idx = []
        for idx in range(len(generated_tokens)):
            eos_idx = torch.where(generated_tokens[idx] == self.eos_id_dict[self.args.model_name])[0] # 返回tuple, [0]是该元素出现位置的tensor
            if len(eos_idx) == 0: # 没有eos_token
                end_idx.append(text_len)
            else:
                end_idx.append(eos_idx[0].item()) # eos_token出现的第一个位置
        return end_idx
    
    def get_generated_tokens_probs_entropy(self, scores, generated_tokens, bt_size):
        top_indices = [] # 存储概率最大的token_id
        top_scores = [] # 存储对应的probs
        ans_scores = [] # 存储seqs对应probs
        ans_entropy = []
        for idx in range(len(scores)): # 遍历每个token
            probs = nn.Softmax(dim=1)(scores[idx]) # batch_size, vocab_size
            tmp_scores, tmp_indices = torch.max(probs, dim=1) # batch_size
            cur_scores = [probs[t, generated_tokens[t, idx]] for t in range(bt_size)] # batch_size, 每个生成token的概率
            cur_entropy = torch.sum(-(probs * torch.log2(probs)), dim=1) # batch_size

            # 当且仅当使用greedy search时, ans_scores = top_scores
            ans_scores.append(cur_scores) # seq_len, batch_size
            ans_entropy.append(cur_entropy.tolist())
            top_indices.append(tmp_indices.tolist())
            top_scores.append(tmp_scores.tolist())
        
        top_indices = torch.tensor(top_indices, dtype=torch.int64).t()
        top_scores = torch.tensor(top_scores).t() # batch_size, text_len
        ans_scores = torch.tensor(ans_scores).t()
        ans_entropy = torch.tensor(ans_entropy).t()
        return top_indices, top_scores, ans_scores, ans_entropy
    
    def get_confidence_idx(self, outs, inputs, end_idx):
        batch_size, input_len = inputs.shape
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        new_token_ids = seqs[:, input_len:]

        pattern = ['certain', 'uncertain', 'ġcertain', 'ġuncertain', '▁certain', '▁uncertain', '*certain', '*uncertain']
        res_idx = []
        for bt in range(len(new_token_ids)):
            bt_res = self.tokenizer.convert_ids_to_tokens(new_token_ids[bt][:end_idx[bt]])
            flag = 0
            for span_len in [3, 2, 1]:
                span_start = 0
                span_end = span_start + span_len - 1
                while span_end < len(bt_res):
                    span_text = ''.join(bt_res[span_start:span_end + 1]).lower().strip()
                    if span_text in pattern:
                        res_idx.append(span_start)
                        flag = 1
                        break   
                    span_start += 1
                    span_end += 1
                if flag == 1:
                    break
            if flag == 0:
                if self.args.model_name in ['llama2-7b-chat', 'llama2-13b-chat']:
                    res_idx.append(1)
                else:
                    res_idx.append(0)
        return res_idx