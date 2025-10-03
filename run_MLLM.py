import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.prompt import get_prompt
from utils.data import VQADataset, QADataset, QADataset_ptb, VQADataset_disc, VQADataset_ptb
from utils.llm import ApiGenerator
from utils.utils import write_jsonl


ra_dict = {
    'none': 'none',
    'sparse': {'sparse_ctxs': 1},
    'dense': {'dense_ctxs': 1},
    'chatgpt': {'gen_ctxs': 100},
    'sparse+dense': {'dense_ctxs': 5, 'sparse_ctxs': 5},
    'gold': {'gold_ctxs': 1},
    'strong': {'strong_ctxs': 10},
    'weak': {'weak_ctxs': 10},
    'rand': {'rand_ctxs': 10},
    'dpr': {'dpr_ctx': 1},
    'extract': {'dpr_ctx': 1},
    'dpr_wrong': {'dpr_ctx_wrong': 1}
}


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/source/nq.json')
    parser.add_argument('--relative_prefix', type=str, default='')
    parser.add_argument('--response', type=str, default='')
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--local_image', type=bool, default=False)
    parser.add_argument('--type', type=str, default='vqa')
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')   
    parser.add_argument('--idx', type=str, default="")   
    parser.add_argument('--model_path', type=str, default="") 
    parser.add_argument('--batch_size', type=int, default=16)   
    parser.add_argument('--task', type=str, default='nq')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--hidden_states', type=bool, default=False)
    parser.add_argument('--output_states', type=bool, default=False)
    parser.add_argument('--attn_weights', type=bool, default=False)
    parser.add_argument('--hidden_idx_mode', type=str, default='last')
    parser.add_argument('--need_layers', type=str, default='last', choices=['all', 'last', 'mid'])
    parser.add_argument('--use_api', type=bool, default=False) # whether to use a model api
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--answer_match_model', type=str, default='') # whether to use a model to match answer,if not, using answer containing.
    parser.add_argument('--answer_match_model_api', type=bool, default=False)
    parser.add_argument('--start_line', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='llm')
    parser.add_argument('--using_consistency', type=bool, default=False)
    parser.add_argument('--consistency_perturb', type=bool, default=False)
    parser.add_argument('--consistency_num', type=int, default=1)
    parser.add_argument('--num_q', type=int, default=10)
    parser.add_argument('--answer_judge', type=str, default='in_answer')
    parser.add_argument('--describe_img', type=bool, default=False)
    parser.add_argument('--description_path', type=str, default='')
    parser.add_argument('--image_noise', type=bool, default=False)
    parser.add_argument('--image_noise_start', type=int, default=0)
    parser.add_argument('--image_noise_step', type=int, default=22)
    parser.add_argument('--multi_step_type', type=str, default=None)
    parser.add_argument('--stream_output', type=bool, default=False)
    parser.add_argument('--logprobs', type=bool, default=False)
    args = parser.parse_args()
    args.ra = ra_dict[args.ra]
    if not args.model_name:
        args.model_name = args.model_path.split('/')[-1].replace('_', '-').lower()

    return args


def main():

    args = get_args()
    print(args)
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        with open("newfile.txt", "w") as f:
            pass
        outfile = open(args.outfile, 'w', encoding='utf-8')
    max_len = None
    if args.model_type == 'mllm' and not args.type == 'vqa_disc':
        if args.using_consistency and args.consistency_perturb:
            all_data = VQADataset_ptb(args, max_len)
        else:
            all_data = VQADataset(args, max_len)
    elif args.model_type == 'llm' and not args.type == 'vqa_disc':
        if args.using_consistency and args.consistency_perturb:
            all_data = QADataset_ptb(args, max_len)
        else:
            all_data = QADataset(args, max_len)
    
    if args.type == 'vqa_disc':
        all_data = VQADataset_disc(args, max_len)
    engine = ApiGenerator(args)
    engine.load_data(all_data)
    res, score = engine.get_res()
    # write_jsonl(res, args.outfile)


if __name__ == '__main__':
    main()
