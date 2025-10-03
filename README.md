# LVLM-Knowledge-Boundary-Perception
This is the official repo for EMNLP 2025 "Do LVLMs Know What They Know? A Systematic Study of Knowledge Boundary Perception in LVLMs".
# Basic Usage
To get desired responses, you need to run ```inference.sh```.
```
python run_MLLM.py \
    --source /data/vqa_500.jsonl \  # question path
    --local_image True \  # whether the image is localy stored or on a website
    --relative_prefix /home/gomall/work/MLLM/data/mllm_datasets/Dyn-QA/ \  # local image relative prefix
    --type qa_short  \  # prompt used, please refer to /utils/prompt.py to see details
    --outfile /res/res.jsonl \ # output res file
    --batch_size 16 \
    --model_name Qwen2.5-7B-Instruct  \  # model name
    --model_type llm \ # if you are using a pure text llm, model_type should be llm, else it should be mllm
    --answer_judge in_answer 
    --start_line 1 \  # write result start from the first line of the output file
    --using_consistency True --consistency_num 20 \ # consistency sampling
    --stream_output True \
    --consistency_perturb True \ # use rephrasing
    --multi_step_type verb_2s_vanilla_re \ # multistep prompts
    --image_noise True --image_noise_start 0 --image_noise_step 5 \ # noised image method parameters
```
- Note
    - Specify ```--type``` to change the prompt.
    - Specity ```--using_consistency``` to enable or disable the use of consistency sampling.
    - Change ```--multi_step_type``` to decide the second round of dialogue prompting.
