import json
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils.prompt import get_prompt, get_prompt_for_multi_round, get_evaluate_output_prompt, get_prompt_multiq, get_prompt_with_disc
import pandas as pd
import os

class QADataset(Dataset):
    """
    Open-domain generation dataset
    """
    def __init__(self, args, max_len):
        self.args = args
        self.max_len = max_len
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.res = []
        self.get_prompted_data()
        

    def read(self, path):
        vqa_data = []
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        lines = lines[self.args.start_line-1:]
        for line in lines:
            data = json.loads(line)
            if 'info' not in data:
                vqa_data.append(json.loads(line))
        return vqa_data
    
    def get_prompted_data(self):
        cur_len = 0
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                cur_len += 1
                if self.max_len is not None and cur_len>self.max_len:
                    break
                self.idxs.append(idx)
                if self.args.usechat:
                    self.prompts.append(get_prompt_for_multi_round(self.data[idx], self.args)) 
                else:
                    self.prompts.append(get_prompt(self.data[idx], self.args)) 
                    if self.args.multi_step_type is not None:
                        self.res.append(self.data[idx]["Res"])
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        if self.args.multi_step_type is None:
            return self.prompts[index]
        return {'question': self.prompts[index], 'Res':self.res[index]}

class MCDataset(Dataset):
    """
    Multi-choice dataset
    """
    # generate input for the given subject
    def __init__(self, args, subject):
        self.args = args
        self.subject = subject
        self.data = self.read('test')
        self.idxs = range(len(self.data))
        self.dev_data = self.read('dev') if self.args.n_shot != 0 else []
        self.get_choice_count()
        self.prompts = []
        self.get_prompted_data()
    
    def get_choice_count(self):
        all_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.choice_cnt = len(self.data[0]) - 2
        self.choices = all_choices[:self.choice_cnt]


    def read(self, mode='test'):
        mmlu_data = pd.read_csv(os.path.join(self.args.source, self.args.data_mode, self.subject + f"_{mode}.csv"), header=None).to_numpy() # no header
        return mmlu_data
    
    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    
    def format_example(self, data, idx, include_answer=True):
        prompt = data[idx][0] # question
        k = len(data[idx]) - 2 # count of choices
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], data[idx][j+1]) # append each candidate answer
        return prompt
    
    def get_prompted_data(self):
        if self.args.task == 'mmlu':
            self.args.subject = ' about' + self.format_subject(self.subject) 
        else:
            self.args.subject = ''
        for idx in range(len(self.data)):
            question = self.format_example(self.data, idx, include_answer=False)
            prompt = get_prompt({'question': question}, self.args)
            self.prompts.append(prompt)
        for item in self.prompts[:5]:
            print(f'example: {item}')
        prompt_len = []
        for item in self.prompts:
            prompt_len.append(len(item.split(' ')))
        self.avg_len = sum(prompt_len)/len(prompt_len)
        self.max_len = max(prompt_len)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]

class VQADataset(Dataset):
    """
    Dynamic vision QA, using the most recently information.
    """
    def __init__(self, args, max_len):
        self.args = args
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.images = []
        self.res = []
        self.max_len = max_len
        self.get_prompted_data()

    def read(self, path):
        vqa_data = []
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        lines = lines[self.args.start_line-1:]
        for line in lines:
            data = json.loads(line)
            if 'info' not in data:
                if "qa_judging" in self.args.type or self.args.type == 'vqa_feedback':
                    data['image_url'] = self.args.relative_prefix+data['qa_prompt']['image_url']
                else:
                    data['image_url'] = self.args.relative_prefix+data['image_url']
            vqa_data.append(data)
        return vqa_data
    
    def get_prompted_data(self):
        cur_len = 0
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                cur_len += 1
                if self.max_len is not None and cur_len>self.max_len:
                    break
                self.idxs.append(idx)
                if self.args.usechat:
                    self.prompts.append(get_prompt_for_multi_round(self.data[idx], self.args)) 
                else:
                    self.prompts.append(get_prompt(self.data[idx], self.args)) 
                    self.images.append(self.data[idx]["image_url"])
                    if self.args.multi_step_type is not None:
                        self.res.append(self.data[idx]["Res"])
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return {"query":self.prompts[index], "image_url":self.images[index] ,**({"Res": self.res[index]} if not self.args.multi_step_type is None else {})}

class afterwardVQADataset(Dataset):
    """
    Hallucination type classy dataset
    """
    def __init__(self, args, max_len):
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.images = []
        self.args = args
        self.max_len = max_len
        self.get_prompted_data()

    def read(self, path):
        vqa_data = []
        f = open(path, 'r', encoding='utf-8')
        for line in f.readlines():
            vqa_data.append(json.loads(line))
        return vqa_data
    
    def get_prompted_data(self):
        cur_len = 0
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx] and 'Res' in self.data[idx] and not self.data[idx]['has_answer']:
                cur_len += 1
                if self.max_len is not None and cur_len>self.max_len:
                    break
                self.idxs.append(idx)
                self.prompts.append(get_evaluate_output_prompt(self.data[idx]['question'] ,
                                                               self.data[idx]['answer'],
                                                                self.data[idx]['Res'], 
                                                                self.args)) 
                self.images.append(self.data[idx]['qa_prompt']["image_url"])
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return {"query":self.prompts[index], "image_url":self.images[index]}

class QADataset_ptb(Dataset):
    def __init__(self, args, max_len):
        self.args = args
        self.max_len = max_len
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.get_prompted_data()
        self.multiple_questions = []
        

    def read(self, path):
        vqa_data = []
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        lines = lines[self.args.start_line-1:]
        for line in lines:
            vqa_data.append(json.loads(line))
        return vqa_data
    
    def get_prompted_data(self):
        cur_len = 0
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                cur_len += 1
                if self.max_len is not None and cur_len>self.max_len:
                    break
                self.idxs.append(idx)
                if self.args.using_consistency:
                    self.prompts.append(get_prompt_multiq(self.data[idx], self.args))
                else:
                    if self.args.usechat:
                        self.prompts.append(get_prompt_for_multi_round(self.data[idx], self.args)) 
                    else:
                        self.prompts.append(get_prompt(self.data[idx], self.args)) 
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]
    
class VQADataset_disc(Dataset):
    def __init__(self, args, max_len):
        self.max_len = max_len
        self.data = self.read(args.source)
        self.descriptions = self.read(args.description_path)
        self.prompts = []
        self.idxs = []
        self.images = []
        self.args = args
        self.get_prompted_data()
        self.multiple_questions = []
        

    def read(self, path):
        qa_data = []
        f = open(path, 'r', encoding='utf-8')
        for line in f.readlines():
            qa_data.append(json.loads(line))
        return qa_data
    
    def get_prompted_data(self):
        cur_len = 0
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                cur_len += 1
                if self.max_len is not None and cur_len>self.max_len:
                    break
                self.idxs.append(idx)
                self.prompts.append(get_prompt_with_disc(self.data[idx], self.descriptions[idx], self.args))
                self.images.append(self.data[idx]["image_url"])
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return {"query":self.prompts[index], "image_url":self.images[index]}
    
class VQADataset_ptb(Dataset):
    def __init__(self, args, max_len):
        self.args = args
        self.max_len = max_len
        self.images = []
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.get_prompted_data()
        self.multiple_questions = []
        
        

    def read(self, path):
        vqa_data = []
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        lines = lines[self.args.start_line-1:]
        for line in lines:
            data = json.loads(line)
            if 'info' not in data:
                data['image_url'] = self.args.relative_prefix+data['image_url']
            vqa_data.append(data)
        return vqa_data
    
    def get_prompted_data(self):
        cur_len = 0
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                cur_len += 1
                if self.max_len is not None and cur_len>self.max_len:
                    break
                self.idxs.append(idx)
                if self.args.using_consistency:
                    self.prompts.append(get_prompt_multiq(self.data[idx], self.args))
                    self.images.append(self.data[idx]["image_url"])
                else:
                    if self.args.usechat:
                        self.prompts.append(get_prompt_for_multi_round(self.data[idx], self.args)) 
                    else:
                        self.prompts.append(get_prompt(self.data[idx], self.args)) 
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return {"query":self.prompts[index], "image_url":self.images[index]}