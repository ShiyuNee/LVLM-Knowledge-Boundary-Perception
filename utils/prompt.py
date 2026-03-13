import json
def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

prompt_dict = {
    'vqa_test':{
        'none': 'Answer the question based on your internal knowledge and image.If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. For example:\n\nAnswer: <most likely answer>\n\nJudgement : <whether you are certain about your answer, just "certain" or "uncertain">\n\nThe question is: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '',
    },
    'vqa_text_test':{
        'none': 'Answer the question in the image.If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. For example:\n\nAnswer: <most likely answer>\n\nJudgement : <whether you are certain about your answer, just "certain" or "uncertain">\n\nThe question is: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'qa_test':{
        'none': 'Answer the question based on your internal knowledge.If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. For example:\n\nAnswer: <most likely answer>\n\nJudgement : <whether you are certain about your answer, just "certain" or "uncertain">\n\nThe question is: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'plain_qa':{
        'none': '{question}', 
        'ra': 'not avaliable', 
        'tail': '\nAnswer: ',
    },
    'vqa_description':{
        'none': 'Answer the question based on your internal knowledge and the image.\nHere\'s a description of the image may helps you answer the question, feel free to utilize or ignore part of it. After answering the question. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nDescription: {description} \nQuestion: {question}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'qa_judging_fact_cot_pun':{
        'none': 'Determine whether the result in the answer is factually correct. Please reason step by step. If you conclude the answer is factually correct, respond with "Yes" at the end; otherwise, respond with "No". You will be punished if the answer is not correct but you say Yes.\nAnswer: {answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'qa_judging_fact_cot':{
        'none': 'Determine whether the result in the answer is factually correct. Please reason step by step.\nAnswer: {answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'qa_judging_fact_explain':{
        'none': 'Determine whether the result in the answer is factually correct and explain why you give this judgement. If you conclude the answer is factually correct, respond with "Yes" at the end; otherwise, respond with "No".\nAnswer: {answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'vqa_judging_image_cot':{
        'none': 'Evaluate whether the content of the answer aligns with the visual information in the image. Please reason step by step.\nAnswer:{answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'vqa_judging_image_explain':{
        'none': 'Evaluate whether the content of the answer aligns with the visual information in the image and explain why you give this judgement. If aligns,  respond with "Yes" at the end; otherwise, respond with "No".\nAnswer: {answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'qa_judging_fact_explain_com':{
        'none': 'Determine the factuality of the result in the answer, explain why you give this judgement.\nAnswer: {answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'vqa_feedback':{
        'none': 'Question:{question}\nAnswer:{answer}\nPlease review the proposed answer and provide feedback on its correctness.',
        'ra': 'not availiable now.',
        'tail': '\nFeedback:'
    },
    'vqa_judging_image_explain_com':{
        'none': 'Evaluate the content alignment between the answer and the visual information in the image, explain why you give this evaluation.\nAnswer: {answer}',
        'ra': 'not avaliable now.',
        'tail': '\nYourResponse '
    },
    'vqa_verbalized_conf_text_vanilla_qimg':{
        'none':'Answer the question in the image based on your internal knowledge. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.',
        'ra':'not avaliable now',
        'tail':'\nYourResponse',
    },
    'vqa_verbalized_conf_text_explain_ins':{
        'none':'Answer the question based on your internal knowledge and the image, explain why you give this answer. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. Remember to follow the "Instruction" below the image.\nQuestion: {question}{paras}{prediction}',
        'ra':'not availiable',
        'tail':'\nAnswer:',
    },
    'vqa_verbalized_conf_text_punish_ins':{
        'none':'Answer the question based on your internal knowledge and the image, explain why you give this answer. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. Remember to follow the "Instruction" below the image.\nQuestion: {question}{paras}{prediction}',
        'ra':'not availiable',
        'tail':'\nAnswer:',
    },
    'vqa_verbalized_conf_text_vanilla_ins':{
        'none':'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. Remember to follow the "Instruction" below the image.\nQuestion: {question}{paras}{prediction}',
        'ra':'not availiable',
        'tail':'\nAnswer:',
    },
    'vqa_text_qimg':{
        'none':'Answer the question in the image based on your internal knowledge.',
        'ra':'not avaliable now',
        'tail':'\nAnswer:',
    },
    'vqa_verbalized_conf_sure': {
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "sure" after the answer. If you are not confident with the answer, please say "unsure" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_conf': {
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "confident" after the answer. If you are not confident with the answer, please say "unconfident" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'qa_verbalized_conf_vanilla':{
        'none': 'Answer the following quesion based on your internal knowledge. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: '
    },
    'vqa_cot':{
        'none': 'Read the question and the image, let\'s analyze step by step, then provide the final answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: '
    }, 
    'vqa_img':{
        'none': 'Answer the question based on your internal knowledge and the image, please first describe the image, then answer the question.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: '
    },
    'vqa_exp':{
        'none': 'Answer the question based on your internal knowledge and the image, explain why you give this answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_punish':{
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. You will be punished if the answer is not right but you say "certain".\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'qa_verbalized_conf_punish':{
        'none': 'Answer the question based on your internal knowledge. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. You will be punished if the answer is not right but you say "certain".\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_cotpun':{
        'none': 'Answer the question based on your internal knowledge and the image. Analyse step by step. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. You will be punished if the answer is not right but you say "certain".\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_idt':{
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. I don\'t think your answer is right.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_conserv':{
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. The question is hard, please be conservative.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_allpun':{
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. You will be PUNISHED if you say "certain".\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_explain':{
        'none': 'Answer the question based on your internal knowledge and the image, explain why you give this answer. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'qa_verbalized_conf_explain':{
        'none': 'Answer the question based on your internal knowledge, explain why you give this answer. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_exppun':{
        'none': 'Answer the question based on your internal knowledge and the image, explain why you give this answer. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer. You will be punished if the answer is not right but you say "certain".\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_vanilla': {
        'none': 'Answer the question based on your internal knowledge and the image. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_prob_vanilla': {
        'none': 'Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer and Probability: ',
    },
    'vqa_verbalized_prob_punish': {
        'none': 'Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question, you will be punished if the probability is high but the guess is wrong. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer and Probability: ',
    },
    'vqa_verbalized_conf_cotimg': {
        'none': 'Answer the question based on your internal knowledge and the image. Analyse step by step. First, describe the image, then steply answer the question. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_conf_cot': {
        'none': 'Answer the question based on your internal knowledge and the image. Analyse step by step. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_prob_cot': {
        'none': 'Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess and probability with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_verbalized_prob_cotimg': {
        'none': 'Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Before giving your answer, first describe the image, then provide a step-by-step explanation of your thought process. Then on a new line give the guess and probability with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence describe the image and step-by-step explanation of your thought process>\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'qa_verbalized_conf_cot': {
        'none': 'Answer the question based on your internal knowledge. Analyse step by step. If you are sure the answer is accurate and correct, please say "certain" after the answer. If you are not confident with the answer, please say "uncertain" after the answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa': {
        'none': 'Answer the question based on your internal knowledge and the image.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'vqa_simple': {
        'none': 'Answer the question based on your internal knowledge and the image with one or a few words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'not avaliable now.',
        'tail': '\nAnswer: ',
    },
    'q_gen':{
        'none': 'Based on the Following question, generate {number} semantically equivalent questions. your output should be a list of strings and add a sequnce number with a dot at the start of each output question, like[1."question1",2."question2",...].\nQuestion: {question}',
        'ra': 'not avaliable now',
        'tail': '\nSemantic equivalent questions:'
    },
    'choice_gen':{
        'none': 'Based on the following question and answer, generat 1 ALTERNATIVE answer in one or a few words.\nQuestion:{question}\nAnswer:{answer}',
        'ra': 'not avaliable now',
        'tail': 'Different answers:'
    },
    'q_gen_image':{
        'none': 'Here is image, question and statement (The statement may be wrong) about a VQA assignment, you should generate {number}semantically equavalent questions\
            based on the Image,original question and answers, your output should be a list, like:[question1, question2,...].\
            You should not output any other things\
            Question: {question}\nStatement: {statement}',
        'ra': 'not availiable now.',
        'tail': '\nsemantic equavalent questions:'
    },
    # 'image_disc':{
    #     'none': 'Describe the image from the "What","Where","When","Who","How" five aspects, express each aspect in one or a few words.',
    #     'ra': 'not availiable now.', 
    #     'tail': '\nYour description:'
    # },
    'image_disc_question':{
        'none': 'Based on the question ,provide information about the image that you think is relevant with the queston. Formulate your response to "What","Where","When","Who","How" five aspects in one or few words.\nQustion: {question}',
        'ra': 'not availiable now.', 
        'tail': '\nYour description:'
    },
    'eval': {
        'none': 'Question: {question}\nGround truth answer: {answer}\nMLLM Statement: {state}',
        'ra': 'not avaliable now', 
        'tail': '\nYour Judgement:',
    },
    'qa': {
        'none': 'Answer the following question based on your internal knowledge.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_short': {
        'none': 'Answer the following question based on your internal knowledge with one or a few words. \nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_cot': {
        'none': 'Answer the question by briefly explaining your reasoning with one or few sentences, then provide the final answer.\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '\nAnswer: ',
    },
    'qa_more': {
        'none': 'Generate 10 possible answers for the following question, each separated by a semicolon. These 10 answers must be different, and your response should be as concise as possible, with no irrelevant words beyond the answers.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    }, 
    'qa_extract': {
        'none': 'Here is the response generated by the model for this question. The response includes reasoning steps, making it difficult to locate the answer. Please extract the answer from the response without generating any other unrelated words. Do not generate conversational words.\nQuestion: {question}\nResponse: {prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nYour extracted answer: ',
    }, 
    'qa_prior': {
        'none': 'Please determine if you can accurately provide the answer to the question. If yes, reply with "certain", otherwise reply with "uncertain". Start your response with "certain" or "uncertain" and do not give any other words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nResponse: ',
    }, 
    'qa_post': {
        'none': 'Here is a question and the answer you provided. Please determine whether the answer is correct. If it is correct, respond with "certain." If it is incorrect, respond with "uncertain." Start your response with "certain" or "uncertain" and do not give any other words.\nQuestion: {question}{paras}{prediction}\nResponse: {prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nResponse: ',
    }, 
    'mc_qa': {
        'none': 'The following are multiple choice questions (with answers){subject}. Select the correct answer without any irrelevant words. Do not include conversational words and do not provide any explanation.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'mc_qa_prior': {
        'none': 'Please determine if you can accurately select the correct answer to the question. If yes, reply with "certain", otherwise reply with "uncertain". Start your response with "certain" or "uncertain" and do not give any other words.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nResponse: ',
    }, 
    'mc_qa_post': {
        'none': 'Here is a question and the answer you select. Please determine whether the answer is correct. If it is correct, respond with "certain." If it is incorrect, respond with "uncertain." Start your response with "certain" or "uncertain" and do not give any other words.\nQuestion: {question}{paras}{prediction}\nResponse: {prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nResponse: ',
    }, 
    'mc_qa_cot': {
        'none': 'The following are multiple choice questions (with answers){subject}. Briefly explain your reasoning with one or few sentences and choose the correct answer. Start with “So, the correct answer is” to select the correct answer.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'mc_qa_evidence': {
        'none': 'The following are multiple choice questions (with answers){subject}. Select the correct answer without any irrelevant words and explain why you choose this answer briefly.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_evidence': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer briefly.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
}

model_template_dict = {
    'qwen-vl-plus-latest':{
        'prefix':'',
        'end':''
    },
    'qwen-omni-turbo':{
        'prefix': '',
        'end': ''
    },
    'llama2-7b-chat':{
        'prefix': '<s>[INST] <<SYS>>\nYou are a helpful assistant<</SYS>>',
        'end': '[/INST]'
    },
    'llama3-8b-instruct':{
        'prefix': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for answering factual questions<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n',
        'end': "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    'qwen2-7b-instruct':{
        'prefix': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'end': '<|im_end|>\n<|im_start|>assistant'
    },
    'llama2-13b-chat':{
        'prefix': '<s>[INST] <<SYS>>\nYou are a helpful assistant<</SYS>>\n\n',
        'end': '[/INST]'
    },
}

model_template_dict_for_multi_round = {
    'llama3-8b-instruct':{
        'sys_prefix': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for answering factual questions',
        'user_prefix': '<|start_header_id|>user<|end_header_id|>\n\n',
        'assis_prefix': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'end': "<|eot_id|>\n"
    },
    'qwen2-7b-instruct':{
        'sys_prefix': '<|im_start|>system\nYou are a helpful assistant.',
        'user_prefix': '<|im_start|>user\n',
        'assis_prefix': '<|im_start|>assistant\n',
        'end': "<|im_end|>\n"
    },
}
def get_prompt(sample, args):
    paras = ""
    ref_key = 'question'
    prompt = prompt_dict[args.type]['none'] # prior
    if args.ra != 'none':  # RA enabled
        ra_dict = args.ra
        i = 0
        doc = []
        for k, v in ra_dict.items():
            v = min(v, len(sample[k]))
            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                i += 1
        paras = '\n'.join(doc)
        prompt = prompt_dict[args.type]['ra']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prediction = sample['Res'] if 'post' in args.type else ""
    # mmlu基础模板里带subject
    # print(args.task)
    if args.task == 'mmlu' or args.task == 'tq':
        prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction, subject=args.subject) + tail
    elif args.task == 'image_disc' or 'qimg' in args.type:
        prompt = prompt + tail
    elif args.task == 'image_disc_question':
        prompt = prompt.format(question=sample[ref_key]) + tail
    elif args.task == 'q_gen':
        prompt = prompt.format(number=args.num_q, question=sample[ref_key])
    elif 'qa_judging' in args.type:
        prompt = prompt.format(answer=sample['Res']) + tail
    elif args.type == 'vqa_feedback' or args.type == 'choice_gen':
        prompt = prompt.format(question=sample['question'], answer=sample['Res'])+tail
    else:
        prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction) + tail
    # 每个模型特有的prompt格式
    if args.model_name in model_template_dict.keys():
        template_prompt = model_template_dict[args.model_name] 
    else:
        template_prompt = {}
        template_prompt['prefix']=''
        template_prompt['end']=''
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt

def get_prompt_for_multi_round(sample, args):
    # question, answer, generate, 10answers
    """
    两轮(post):
        - factual question
        - response
        - determine right
    三轮(post_multi_round):
        - factual question
        - response
        - generate 10 answers
        - 10 answers
        - determine right
    """
    prompt = ''
    template_prompt = model_template_dict_for_multi_round[args.model_name]
    # sys
    prompt += template_prompt['sys_prefix']
    prompt += template_prompt['end']
    if args.type == 'qa_post':
        sample['question'] = sample['question'][:2] # qa_post_multi_round需要4个元素, qa_post只需要两个元素
    for idx in range(len(sample['question'])):
        if idx % 2 == 0:
            # question
            prompt += template_prompt['user_prefix']
            prompt += sample['question'][idx]
            prompt += template_prompt['end']
        else:
            # answer
            prompt += template_prompt['assis_prefix']
            prompt += sample['question'][idx]
            prompt += template_prompt['end']
   
    prompt += template_prompt['user_prefix']
    prompt += f'Please determine whether your response [{sample["question"][1]}] contains the correct answer. If yes, respond with "certain." If it is incorrect, respond with "uncertain." Start your response with "certain" or "uncertain" and do not give any other words.'
    prompt += template_prompt['end']
    prompt += template_prompt['assis_prefix']
    # print(prompt)
    return prompt

def get_evaluate_output_prompt(question ,answers, state, args):
    prompt = prompt_dict['eval']['none'] # prior
    tail = prompt_dict['eval']['tail'] if not args.usechat else ""
    # print(args.task)
    prompt = prompt.format(question=question, answer=str(answers), state=state) + tail
    # 每个模型特有的prompt格式
    template_prompt = model_template_dict[args.model_name]
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt

def get_prompt_multiq(sample, args):
    questions = sample['multiple_questions']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prompts = []
    for question in questions:
        prompt = prompt_dict[args.type]['none']
        prompt = prompt.format(question=question, paras='', prediction='') + tail
        if args.model_name in model_template_dict.keys():
            template_prompt = model_template_dict[args.model_name] 
        else:
            template_prompt = {}
            template_prompt['prefix']=''
            template_prompt['end']=''
        prompt = template_prompt['prefix'] + prompt + template_prompt['end']
        prompts.append(prompt)
    return prompts
def get_prompt_with_disc(sample, description, args):
    question = sample['question']
    prompt = prompt_dict['vqa_description']['none']
    tail = prompt_dict['vqa_description']['tail']
    prompt = prompt.format(question=question, description=description['Res']) + tail
    if args.model_name in model_template_dict.keys():
        template_prompt = model_template_dict[args.model_name] 
    else:
        template_prompt = {}
        template_prompt['prefix']=''
        template_prompt['end']=''
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt

if __name__ == '__main__':
    model_name='qwen7b'
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    mode = 'test'
    dataset = 'nq'
    out_path = f'{base_dir}/{dataset}/multi_round/{dataset}_{mode}_{model_name}.jsonl'
    data = read_json(out_path)
    get_prompt_for_multi_round(data[0], {'model_name': 'llama3-8b-instruct'})

