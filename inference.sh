

python /home/gomall/work/MLLM/run_MLLM.py \
    --source /home/gomall/work/MLLM/data/multiple_answers/year_200/office_200/office_6a_200.jsonl \
    --local_image True \  #不用管，运行LVLM时用的
    --relative_prefix /home/gomall/work/MLLM/data/mllm_datasets/Dyn-QA/ \   #不用管
    --type qa_short --ra none --outfile /home/gomall/work/MLLM/multi_answer_res/qwen2.5-7B/year_200/greedy/office_200/office_6a_200_res.jsonl \ #type为提示的prompt，具体在utils文件夹下的prompt.py一般也不用改，就改那个输出路径即可
    --model_path none \ #不用改
    --batch_size 16 --task vqa --hidden_states False --use_api True \  #不用改
    --model_name /home/gomall/models/Qwen2.5-7B-Instruct  \  #模型名称，调api用的。我们通常使用vllm框架运行api，例如python -m vllm.entrypoints.openai.api_server --model /home/gomall/work/llava-1.5-7b-hf --dtype auto --api-key 114514 --tensor_parallel_size 2
    --model_type llm \ # 不用改
    --answer_judge in_answer --answer_match_model facebook/bart-large-mnli \  # 不用改
    --description_path D:/MLLM/LLM-Knowledge-Boundary-Perception-via-Internal-States-master/data/mllm_datasets/Dyn-QA/image_disc.jsonl \ # 不用改
    --start_line 1 \  # 写入文件的第几行，一般不用改。
    --using_consistency True --consistency_num 20 \ # 是否进行consistency 采样，假如不使用的话就把这行注释掉
    --stream_output True # 这里是为了适应有些手写的api运行程序不支持流式输出，一般不用改
    # 下面这些用不到
    # --logprobs True \
    # --consistency_perturb True\
    # --multi_step_type verb_2s_vanilla_re \
    # --image_noise True --image_noise_start 0 --image_noise_step 5 \
    
    
    
    
    
    
    
   
   