"""
Official LLM-based Evaluation Script for AVerImaTeC Pipeline (Standalone Version)
Uses the same evaluation methods as the official AVerImaTeC competition
No dependencies on Reference code - all functions copied here
"""

import json
import os
import sys
import argparse
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ============================================================================
# Utility Functions (copied from Reference/AVerImaTec_Shared_Task/prepare_submission/utils.py)
# ============================================================================

threshold = 9

def compute_scores(score, len_gt_evid, len_val_evid):
    if len_val_evid == 0 or len_gt_evid == 0:
        return 0.0, 0.0, 0.0
    precision = score["pred_in_ref"] / len_val_evid
    recall = score["ref_in_pred"] / len_gt_evid
    if precision < 0:
        precision = 0
    if recall < 0:
        recall = 0
    if recall > 1.:
        recall = 1.
    if precision > 1.:
        precision = 1.
    if precision == 0 and recall == 0:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def justi_recall_compute(raw_gen, score):
    """
    Compute justification recall from LLM feedback
    EXACT MATCH to Reference utils.py lines 26-34
    """
    pred_facts = raw_gen.strip().split('[PRED in REF Exp]: ')[1].split('[REF in PRED]:')[0].strip()
    ref_facts = raw_gen.strip().split('[REF in PRED Exp]:')[-1].strip()

    num_gt = len(re.findall(r'\b\d+\.\s', ref_facts))
    num_pred = len(re.findall(r'\b\d+\.\s', pred_facts))

    _, recall, _ = compute_scores(score, num_gt, num_pred)
    return recall

def ques_recall_compute(score, num_gt, num_pred):
    _, recall, _ = compute_scores(score, num_gt, num_pred)
    return recall

def compute_scores_detail(score, len_gt_evid, len_val_evid):
    if len_val_evid == 0 or len_gt_evid == 0:
        return None, None, None
    precision = score["pred_in_ref"] / len_val_evid
    recall = score["ref_in_pred"] / len_gt_evid
    if precision < 0:
        precision = 0
    if recall < 0:
        recall = 0
    if recall > 1.:
        recall = 1.
    if precision > 1.:
        precision = 1.
    if precision == 0 and recall == 0:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def get_auto_recall(result, image_scores, len_ref, len_pred):
    pred_in_ref = image_scores['pred_in_ref']
    ref_in_pred = image_scores['ref_in_pred']
    pred_dict = defaultdict(int)
    num_pred_in_ref = 0
    for i, info in enumerate(pred_in_ref):
        try:
            pred_idx = int(info['info'][0].split('_')[-1])
            ref_idx = int(info['info'][1].split('_')[-1])
        except:
            continue
        if pred_idx in pred_dict:
            continue
        pred_dict[pred_idx] += 1
        try:
            if int(info['score']) < threshold:
                continue
            else:
                num_pred_in_ref += 1
        except:
            continue
    ref_dict = defaultdict(int)
    num_ref_in_pred = 0
    for i, info in enumerate(ref_in_pred):
        try:
            pred_idx = int(info['info'][1].split('_')[-1])
            ref_idx = int(info['info'][0].split('_')[-1])
        except:
            continue
        if ref_idx in ref_dict:
            continue
        ref_dict[ref_idx] += 1
        try:
            if int(info['score']) < threshold:
                continue
            else:
                num_ref_in_pred += 1
        except:
            continue
    precision, recall, f1 = compute_scores_detail(
        {'ref_in_pred': num_ref_in_pred, 'pred_in_ref': num_pred_in_ref},
        len_ref, len_pred
    )
    return precision, recall, f1

# ============================================================================
# Template Loading Functions (EXACT MATCH to Reference templates)
# ============================================================================

def load_template(template_name):
    """
    Load evaluation template from embedded strings.
    These templates match EXACTLY with Reference/AVerImaTec_Shared_Task/templates/*.txt
    Note: Templates do NOT include [PRED]/[REF] - those are appended by gen_incontext_input_textonly()
    """
    templates = {
        # From Reference/AVerImaTec_Shared_Task/templates/ques_evaluation_text.txt
        # NOTE: Trailing spaces on lines 8, 17, 19 match reference exactly
        'ques_evaluation': """You will get as input a reference question set ([REF]) and a predicted question set ([PRED]).
Please verify the correctness of the predicted questions by comparing it to the reference questions, following these steps:
1. Evaluate each question in the predicted question set individually: Check whether it is covered by any question in the reference set ([REF]). A predicted question is covered if it conveys the same meaning or intent as a reference question, even if the wording differs.
2. Evaluate each question in the reference question set individually: Check whether it is covered by any question in the predicted set ([PRED]), using the same criteria. Do not use additional sources or background knowledge.
3. Finally summarise (1.) Count how many predicted questions are covered by the reference questions and provide explanations([PRED in REF] and [PRED in REF Exp]), (2.) Count how many reference questions are covered by the predicted questions and provide explanations ([REF in PRED] and [REF in PRED Exp]).
Generate the output as shown in the examples below:

[PRED]: 1. Is there a correlation between CO2 levels and climate change? 2. Where and when was the image taken? 3. What is the caption of the chart in the image? 
[REF]: 1. Was the source article publishing the chart discussing climate change? 2. Will a raise of CO2 levels lead to global warming? 3. Which country was shown in the image? 4. When was the image taken?
[PRED in REF]: 2
[PRED in REF Exp]: 1. The question is similar to the second reference question. 2. The question conveys similar information to the third and fourth question in the reference set. 3. The question is not covered by nor similar to any reference question.
[REF in PRED]: 3
[REF in PRED Exp]: 1. The question is not covered by the predicted question set. 2. The question is covered by the first predicted question. 3. The question is covered by the second question of the predicted questions. 4. The question is covered by the second predicted question.

[PRED]: 1. What is the source of the image? 2. Did the U.S. government provide direct financial aid to Ukraine in 2023? 3. What is the publication date of the image? 4. How did the U.S. justify its continued support for Ukraine in 2023?
[REF]: 1. What is the event in the image? 2. When was the image first published? 3. Did the U.S. provide any financial assistance to Ukraine in 2023?
[PRED in REF]: 2 
[PRED in REF Exp]: 1. The questions is not covered by any reference question. 2. The question is covered by the third reference question. 3. The question is covered by the second reference question. 4. No similar questions to this question could be found in the reference question set.
[REF in PRED]: 2 
[REF in PRED Exp]: 1. The question is not covered by any predicted question. 2. The question is covered by the third predicted question. 3. The question is similar to the second predicted question so that it is covered by a predicted question.


Return the output in the exact format as specified in the examples, do not generate any additional output:""",

        # Justification evaluation template
        # NOTE: Trailing spaces on lines 23, 25 match reference exactly
        'justi_evaluation': """You will get as input a reference justification ([REF]) and a predicted justification ([PRED]).
Please verify the correctness of the predicted justifications by comparing it to the reference justifcations, following these steps:
1. Decompose the predicted justification into atomic facts [PRED_FACT]. Each fact should be a separate sentence.
2. Decompose the reference justification into atomic facts [REF_FACT]. Each fact should be a separate sentence.
3. Evaluate each fact in the predicted fact set individually: Check whether it is covered by any fact in the reference set ([REF_FACT]). A predicted fact is covered if it conveys the same meaning or intent as a reference fact, even if the wording differs.
4. Evaluate each fact in the reference fact set individually: Check whether it is covered by any fact in the predicted set ([PRED_FACT]), using the same criteria. Do not use additional sources or background knowledge.
5. Finally summarise (1.) Count how many predicted facts are covered by the reference facts and provide explanations([PRED in REF] and [PRED in REF Exp]), (2.) Count how many reference facts are covered by the predicted facts and provide explanations ([REF in PRED] and [REF in PRED Exp]).
Generate the output as shown in the examples below:

[PRED]: While evidence confirms that Karikó and Weissman won the 2023 Nobel Prize, none of the sources verify that the photo is from the Nobel ceremony, that the two masked people are indeed Karikó and Weissman, or that they wore masks while accepting the award, so the claim cannot be substantiated with the information available.
[REF]: The claim is successfully refuted by proving that the awards were presented on 4/13/22 a year and a half prior to the awarding of the Nobel Prizes.
[PRED_FACT]: 1. Karikó and Weissman won the 2023 Nobel Prize. 2. None of the sources verify that the photo is from the Nobel ceremony. 3. None of the sources verify that the two masked people are indeed Karikó and Weissman, or that they wore masks while accepting the award.
[REF_FACT]: 1. The awards were presented on 4/13/22. 2. 4/13/22 is year and a half prior to the awarding of the Nobel Prizes.
[PRED in REF]: 0
[PRED in REF Exp]: 1. The fact is not covered by the predicted fact set. 2. The fact is not covered by the predicted fact set. 3. The fact is not covered by nor similar to any reference fact.
[REF in PRED]: 0
[REF in PRED Exp]: 1. The fact is not covered by the predicted fact set. 2. The fact is not covered by nor similar to any reference fact.

[PRED]: The claim of flowers blooming in Antarctica is refuted because the provided image shows a location in West Greenland, and the typical Antarctic vegetation consists of non-flowering plants like mosses and lichens. The geographic location and type of flora are inconsistent with the claim.
[REF]: The claim is successfully refuted by proving the origin of the photo is in Greenland and that flowers do not grow in Antarctica.
[PRED_FACT]: 1. The provided image shows a location in West Greenland. 2. The typical Antarctic vegetation consists of non-flowering plants like mosses and lichens.
[REF_FACT]: 1. The origin of the photo is in Greenland. 2. The flowers do not grow in  Antarctica.
[PRED in REF]: 2 
[PRED in REF Exp]: 1. The fact is consistent with the first fact in the reference fact set. 2. The fact is covered by the second fact in the reference set.
[REF in PRED]: 2 
[REF in PRED Exp]: 1. The fact is similar to the first fact in the predicted fact set. 2. The fact is covered by the second fact in the predicted fact set.


Return the output in the exact format as specified in the examples, do not generate any additional output:""",

        # From Reference/AVerImaTec_Shared_Task/templates/evid_evaluation_text_seperate.txt
        'evid_evaluation': """You will get as input a reference evidence ([REF]) and a predicted evidence ([PRED]). [IMG_1], [IMG_2] .. are placeholders for images and they are regared as the same text token.
Please verify the correctness of the predicted evidence by comparing it to the reference evidence. Note, a fact with "no answer could be found .." or "it is unkown .." contradicts with facts mentioning any exact information (i.e., indicating the answer can be found and it is known). Please verify following these steps:
1. Evaluate each fact in the predicted evidence individually: is the fact supported by the REFERENCE evidence (reference evidence presents a similar )? Do not use additional sources or background knowledge.
2. Evaluate each fact in the reference evidence individually: is the fact supported by the PREDICTED evidence? Do not use additional sources or background knowledge.
3. Finally summarise (1.) how many predicted facts are supported by the reference evidence, which refernece evidence supports which predicted facts and explanations([PRED in REF] and [PRED in REF Exp]), (2.) how many reference facts are supported by the predicted evidence, which predicted evidence supports which reference fact and explanations ([REF in PRED] and [REF in PRED Exp]).
Generate the output as shown in the examples below:

[PRED]: 1. The date of [IMG_1] can be decided. 2. Ilan Omar has attended the training in [IMG_2]. 3. It is unknown when the raid in Washington took place.
[REF]: 1. [IMG_1] was taken in Jan. 20, 2003. 2. No evidence can be found related to the type of missle in [IMG_2]. 3. The woman in [IMG_3] for a training is not Ilan Omar. 4. The raid in Washington took place on Saturday, Oct. 26, 1999. 5. Prince Phillip wore the Royal Guard uniform shown in [IMG_4] previously in Jan. 2003. 5. The missle in [IMG_1] is Fateh 110.
[PRED in REF]: 0; None
[PRED in REF Exp]: 1. The fact is similar to the fifth referece fact while it does not mention the type of missle. 2. The fact contradicts with its relevant fact, the third fact, in the evidence set. It fact in the reference set claims the woman in the training is not Ilan Omar. 3. The fact is refuted by the fourth fact in the reference set, which claims the date of the raid in Washington is Oct. 26, 1999.
[REF in PRED]: 0; None
[REF in PRED Exp]: 1. No relevant evidence could be found in the predicted evidence. 2. It is refuted by the first fact in the predicted evidence set. 3. It is contracted with the second fact in the predicted evidence set, which claims the woman in the training is Ilan Omar. 4. It is not supported by any evidence in the prediction set and the third fact in the prediction set contradicts with it. 5. The first fact in the prediction set has similar content to this fact while stating unknown to the type of missle, which contradicts with this fact.

[PRED]: 1. The missle in [IMG_1] is Fateh 110. 2. Ilan Omar has attended the training in [IMG_2]. 3. Prince Phillip wore the Royal Guard uniform in Jan. 14, 2003. 4.The raid in Washington took place on Saturday, Oct. 26, 1999.
[REF]: 1. [IMG_1] was taken in Jan. 20, 2003. 2. No evidence can be found related to the type of missle in [IMG_2]. 3. The woman in [IMG_3] for a training is not Ilan Omar. 4. No answer was found regarding when the raid in Washington took place. 5. Prince Phillip wore the Royal Guard uniform shown in [IMG_4] previously in Jan. 2003.
[PRED in REF]: 1; (PRED_3,REF_5)
[PRED in REF Exp]: 1. No relevant evidence to the fact can be found in the reference evidence set. 2. The fact contradicts with its relevant fact, the third fact, in the evidence set. It fact in the reference set claims the woman in the training is not Ilan Omar. 3. The fact is supported by the fifth fact in the evidence set. 4. The fact is refuted by the fourth fact in the reference set, which claims the date of the raid in Washington is unknown.
[REF in PRED]: 2; (REF_1,RPED_3);(REF_5,PRED_3)
[REF in PRED Exp]: 1. It is supported by the third fact in the predicted evidence. 2. It is refuted by the first fact in the predicted evidence set. 3. It is contracted with the second fact in the predicted evidence set, which claims the woman in the training is Ilan Omar. 4. It is refuted by the fourth fact in the predicted evidence which claims the date of the raid could be found. 5. The fact aligns with the third fact in the predicted evidence set.

[PRED]: 1. A man allegedly called on the HN Reliance Foundation Hospital and issued threats to Mukesh Ambani. 2. Astrologer Chirag Daruwalla issues predictions for Mukesh Ambani. 3. Mukesh Ambani is an Indian industrialist. 4. Mukesh Ambani is the chairman and managing director of Reliance Industries. 5. Mukesh Ambani is Asia's richest man. 6. Mukesh Ambani lost $7 billion from his networth as Reliance Industries Ltd.'s shares tumbled to the lowest price in more than three months.
[REF]: 1. Mukhesh Aambi is the richest man in Asia. 2. On September 5, 2020 a photograph of Mukesh Ambani was taken claiming he had been diagnosed with pancreatic cancer and had undergone surgery. 3. On October 19, 2020 a video of Mukesh Ambani was filmed at the virtual launch of NK Singh's book. 4. On November 2, 2020 a Facebook post was posted confirming that Mukesh Ambani had lost 30 kgs, been diagnosed with pancreatic cancer and had had liver transplant surgery. 5. A photo of Mukhesh Ambani supposedly recieving surgery actually taken in Liechtenstein.
[PRED in REF]: 1; (PRED_5,REF_1) 
[PRED in REF Exp]: 1. A man allegedly called on the HN Reliance Foundation Hospital and issued threats to Mukesh Ambani. The reference evidence does not mention anything about a man calling and threatening Mukesh Ambani. Not enough information. 2. Astrologer Chirag Daruwalla issues predictions for Mukesh Ambani. The reference evidence does not mention anything about an Astrologer giving predictions about Mukesh Ambani's future. Not enough information. 3. Mukesh Ambani is an Indian industrialist. The reference evidence does not mention that Mukesh Ambani is an Indian industrialist. Not enough information. 4. Mukesh Ambani is the chairman and managing director of Reliance Industries. The reference evidence does not mention that Mukesh Ambani is the managing director of Reliance Industries. Not enough information. 5. Mukesh Ambani is Asia's richest man. The fact 'Mukesh Ambani is Asia's richest man' is supported by the reference evidence. 6. Mukesh Ambani lost $7 billion from his networth as Reliance Industries Ltd.'s shares tumbled to the lowest price in more than three months. The reference evidence does not mention that Mukesh Ambani lost money or why he lost it. Not enough information.)
[REF in PRED]: 1; (REF_1,PRED_5) 
[REF in PRED Exp]: 1. Mukhesh Aambi is the richest man in Asia. The predicted evidence mentions that Mukhesh Ambani is Asia's richest man, this fact is hence supported. 2. On September 5, 2020 a photograph of Mukesh Ambani was taken claiming he had been diagnosed with pancreatic cancer and had undergone surgery. The predicted evidence does not mention anything about Mukhesh Ambani's cancer diagnosis or surgery. Not enough information. 3. On October 19, 2020 a video of Mukesh Ambani was filmed at the virtual launch of NK Singh's book. Predicted evidence does not mention Ambani attending any book launch. Not enough information. 4. On November 2, 2020 a Facebook post was posted confirming that Mukesh Ambani had lost 30 kgs, been diagnosed with pancreatic cancer and had had liver transplant surgery. The predicted evidence does not mention any of this. Not enough information. 5. A photo of Mukhesh Ambani supposedly recieving surgery was actually taken in Liechtenstein. The predicted evdience does not mention anything about a survey or Ambani being in Liechtenstein. Not enough information.

[PRED]: 1. No answer could be found about the type of missle in [IMG_1]. 2. Ilan Omar has attended the training in [IMG_2]. 3. It is unknown when the raid in Washington took place.
[REF]: 1. [IMG_1] was taken in Jan. 20, 2003. 2. No evidence can be found related to the type of missle in [IMG_2]. 3. The woman in [IMG_3] for a training is not Ilan Omar. 4. The raid in Washington took place on Saturday, Oct. 26, 1999. 5. Prince Phillip wore the Royal Guard uniform shown in [IMG_4] previously in Jan. 2003. 5. The missle in [IMG_1] is Fateh 110.
[PRED in REF]: 0; None
[PRED in REF Exp]: 1. The fact is similar to the fifth referece fact while it does not mention the type of missle. 2. The fact contradicts with its relevant fact, the third fact, in the evidence set. It fact in the reference set claims the woman in the training is not Ilan Omar. 3. The fact is refuted by the fourth fact in the reference set, which claims the date of the raid in Washington is Oct. 26, 1999.
[REF in PRED]: 0; None
[REF in PRED Exp]: 1. No relevant evidence could be found in the predicted evidence. 2. It is refuted by the first fact in the predicted evidence set. 3. It is contracted with the second fact in the predicted evidence set, which claims the woman in the training is Ilan Omar. 4. It is not supported by any evidence in the prediction set and the third fact in the prediction set contradicts with it. 5. The first fact in the prediction set has similar content to this fact while stating unknown to the type of missle, which contradicts with this fact.

[PRED]: 1. [IMG_1] was taken on Jan. 19, 2025. 2. The current view of the benches in [IMG_2] is [IMG_3]. 3. The date of the claim is Nov. 22, 2023. 
[REF]: 1. The claim was made on Jan. 22, 2021. 2. [IMG_2] was taken on Jan. 19, 2025. 3. The benches in [IMG_1] currently look like [IMG_3]. 4.Trump dressed as [IMG_1] in the meeting.
[PRED in REF]: 2; (PRED_1,REF_2);(PRED_2,REF_3)
[PRED in REF Exp]: 1. The second piece of evidence in the reference evidence supports it. 2. The third evidence in the evidence set has a similar meaning to this fact. 3. The fact claims the date as Nov. 22, 2023, which is different from the first fact in the refence evidence, Jan. 22, 2021.
[REF in PRED]: 2; (REF_2,PRED_1);(REF_3,PRED_2)
[REF in PRED Exp]: 1. The fact claims the date as Jan. 22, 2021, which is different from the third fact in the predicted evidence, Nov. 22, 2023. 2. It is supported by the first fact in the predicted evidence. 3. It is supported by the second fact in the predicted evidence. 4. No related facts can be found in the predicted evidence set.


Return the output in the exact format as specified in the examples, do not generate any additional output:"""
    }
    return templates.get(template_name, "")

def gen_incontext_input_textonly(pred, ref, demos):
    """
    Generate input for LLM evaluation - EXACT MATCH to Reference ref_eval.py line 39-45
    Appends [PRED] and [REF] sections after the template (demos)
    """
    texts = []
    texts.append(demos)
    texts.append("\n[PRED]: " + pred)
    texts.append("[REF]: " + ref)
    texts = '\n'.join(texts)
    return texts

# ============================================================================
# Score Extraction
# ============================================================================

def score_extraction(feedback):
    """Extract scores from LLM feedback - matches reference implementation"""
    pred_in_ref = feedback.split('[PRED in REF]: ')[-1].split('\n')[0].split(';')[0].strip()
    ref_in_pred = feedback.split('[REF in PRED]: ')[-1].split('\n')[0].split(';')[0].strip()

    if pred_in_ref.isdigit():
        pred_in_ref = int(pred_in_ref)
    else:
        pred_in_ref = 0
    if ref_in_pred.isdigit():
        ref_in_pred = int(ref_in_pred)
    else:
        ref_in_pred = 0

    score = {
        'ref_in_pred': ref_in_pred,
        'pred_in_ref': pred_in_ref
    }

    # Extract detailed matching information for image scoring
    if len(feedback.split('[PRED in REF]: ')[-1].split('\n')[0].split(';')):
        score['detailed_ref_in_pred'] = ';'.join(feedback.split('[REF in PRED]: ')[-1].split('\n')[0].split(';')[1:]).strip()
        score['detailed_pred_in_ref'] = ';'.join(feedback.split('[PRED in REF]: ')[-1].split('\n')[0].split(';')[1:]).strip()

    return score

# ============================================================================
# QA to Evidence Conversion
# ============================================================================

def gen_incontext_input_qa(ques, ans, demos):
    """
    Generate input for QA to evidence conversion
    EXACT MATCH to Reference qa_to_evidence.py gen_incontext_input() lines 5-12
    """
    texts = []
    texts.append(demos)
    texts.append("[QUES]: " + ques)
    texts.append("[ANS]: " + ans)
    texts.append("[STAT]:")
    texts = '\n'.join(texts)
    return texts

def qa_to_evid(question, answer, llm, llm_name):
    """
    Convert QA pair to evidence statement using LLM
    EXACT MATCH to Reference qa_to_evidence.py qa_to_evid() lines 14-44
    """
    # EXACT COPY from Reference/AVerImaTec_Shared_Task/templates/qa_to_evid_demos.txt
    demonstrations = """You are a expert writer. Given a question ([QUES]) and its answer [ANS], your goal is to convert the QA pair into a statement [STAT]. There could be images either in the question or the answer, which we use special tokens [IMG_1], [IMG_2] ... as placeholders for images. For instance, the question "When was the image published? [IMG]" asks for the publication date of the image denoted as [IMG]. Below are some examples:


[QUES]: What is the date of the claim?
[ANS]: Nov. 22, 2023.
[STAT]: The date of the claim is Nov. 22, 2023.

[QUES]: Did Trump pretended to be the palace guard in the meeting?
[ANS]: [IMG_1]
[STAT]: Trump dressed as [IMG_1] in the meeting.

[QUES]: When was the image shot? [IMG_1]
[ANS]: The image has been taken on Jan. 25, 1998.
[STAT]: [IMG_1] was taken on Jan. 25, 1998.

[QUES]: What is the current view of the benches in the image? [IMG_1]
[ANS]: [IMG_2]
[STAT]: The current view of the benches in [IMG_1] is [IMG_2].

[QUES]: What were the missile systems deployed by Iran in 2018?
[ANS]: [IMG_1], [IMG_2]
[STAT]: The missile systems deployed by Iran in 2018 were [IMG_1], [IMG_2].

[QUES]: What is the name of the temple? [IMG_1]
[ANS]: Bright Hill Temple.
[STAT]: The temple in [IMG_1] is Bright Hill Temple.

[QUES]: Can onions absorb illness from a person's feet?
[ANS]: No answer could be found.
[STAT]: No answer was found regarding whether onions can absorb illness from a person's feet.

[QUES]: Were there any changes to the park? [IMG_1]
[ANS]: [IMG_2], [IMG_3]
[STAT]: The park in [IMG_1] has been changed to [IMG2], [IMG_3].


Please convert the QA pair below into its statement: """

    incontext_input = gen_incontext_input_qa(question, answer, demonstrations)

    if 'gemini' in llm_name:
        response = llm.models.generate_content(
            model=llm_name,
            contents=incontext_input
        )
        statement = response.text.replace('[STAT]:', '').strip()
        return statement
    elif 'gemma' in llm_name:
        messages = [{"role": "user", "content": [{'type': 'text', 'text': incontext_input}]}]
        inputs = llm["processor"].apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(llm['model'].device)
        with torch.no_grad():
            generated_ids = llm['model'].generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = llm["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        statement = response.replace('[STAT]:', '').strip()
        return statement

    return f"{question} {answer}"

def convert_qa_format(question_info, llm, llm_name, image_dir):
    """Convert QA format to evidence format"""
    answers = question_info["answers"]
    ques_txt = question_info['question'].replace('\n', '; ')
    related_images = []
    ques_img_str = []
    ans_text = []
    
    if len(question_info.get('input_images', [])):
        rel_images = question_info["input_images"]
        for image in rel_images:
            img_path = os.path.join(image_dir, image)
            if os.path.exists(img_path):
                related_images.append(img_path)
                ques_img_str.append(f'[IMG_{len(related_images)}]')
    
    for j, answer in enumerate(answers):
        answer_type = answer["answer_type"]
        if answer_type == 'Image':
            image_answers = answer.get("image_answers", [])
            for image in image_answers:
                img_path = os.path.join(image_dir, image)
                if os.path.exists(img_path):
                    related_images.append(img_path)
                    ans_text.append(f'[IMG_{len(related_images)}]')
        else:
            ans_text.append(answer.get("answer_text", ""))
        if answer_type == 'Boolean':
            boolean_explanation = answer.get("boolean_explanation", "")
            ans_text.append(boolean_explanation)
    
    ans_text = ' '.join(ans_text).replace('\n', '; ')
    if len(ques_img_str):
        evid_ques = ques_txt + ', '.join(ques_img_str)
    else:
        evid_ques = ques_txt
    
    evid = qa_to_evid(evid_ques, ans_text, llm, llm_name)
    
    evid_info = {
        'text': evid,
        'images': related_images
    }
    return evid_info

# ============================================================================
# Evaluation Functions (copied from ref_eval.py)
# ============================================================================

def textual_val_single(ref, pred, path, eval_name, model, eval_type="", debug_mode=False):
    """
    Evaluate textual content (questions or justifications)
    EXACT MATCH to ref_eval.py textual_val_single() lines 370-408
    eval_type: 'question' or 'justification'
    path: not used but kept for compatibility
    """
    if eval_type == 'justification':
        val_demo = load_template('justi_evaluation')
    elif eval_type == 'question':
        val_demo = load_template('ques_evaluation')
        # EXACT MATCH to ref_eval.py lines 378-382
        if pred[0][0].isdigit() == False:
            pred = [str(k+1) + '. ' + row for k, row in enumerate(pred)]
        pred = ' '.join(pred)
        ref = [str(k+1) + '. ' + row for k, row in enumerate(ref)]
        ref = ' '.join(ref)

    incontext_input = gen_incontext_input_textonly(pred, ref, val_demo)

    if 'gemini' in eval_name:
        response = model.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=incontext_input
        )
        feedback = response.text
    elif 'gemma' in eval_name:
        messages = [
            {"role": "user", "content": [{'type': 'text', 'text': incontext_input}]}
        ]
        inputs = model["processor"].apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model['model'].device)
        with torch.no_grad():
            generated_ids = model['model'].generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        feedback = model["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]

    processed_score = score_extraction(feedback)
    return feedback, processed_score

def val_evid_idv(model, model_name, pred_evid, ref_evid, text_val, seperate_val):
    """
    Evaluate evidence
    EXACT MATCH to Reference ref_eval.py val_evid_idv() lines 181-230
    """
    pred = [str(k+1) + '. ' + row['text'] for k, row in enumerate(pred_evid)]
    gt = [str(k+1) + '. ' + row['text'] for k, row in enumerate(ref_evid)]
    ref = '. '.join(gt)
    pred = '. '.join(pred)

    if text_val or seperate_val:
        if seperate_val:
            template = load_template('evid_evaluation')  # seperate_val_demo
        else:
            template = load_template('evid_evaluation')  # text_val_demo (same template for now)
        incontext_input = gen_incontext_input_textonly(pred, ref, template)

        if 'gemini' in model_name:
            response = model.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=incontext_input
            )
            feedback = response.text
        elif 'gemma' in model_name:
            messages = [
                {"role": "user", "content": [{'type': 'text', 'text': incontext_input}]}
            ]
            inputs = model["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model['model'].device)
            with torch.no_grad():
                generated_ids = model['model'].generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            feedback = model["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
    else:
        # joint image-text evaluation path (not used in our pipeline currently)
        template = load_template('evid_evaluation')
        incontext_input = gen_incontext_input_textonly(pred, ref, template)
        if 'gemini' in model_name:
            response = model.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=incontext_input
            )
            feedback = response.text
        elif 'gemma' in model_name:
            messages = [{"role": "user", "content": [{'type': 'text', 'text': incontext_input}]}]
            inputs = model["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model['model'].device)
            with torch.no_grad():
                generated_ids = model['model'].generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            feedback = model["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    processed_score = score_extraction(feedback)
    return feedback, processed_score

def compute_image_scores(model, model_name, pred_evid, ref_evid, evid_val_score):
    """
    Compute image similarity scores using LLM
    EXACT MATCH to Reference ref_eval.py compute_image_scores() lines 232-290
    """
    from PIL import Image

    prompt = "Given two sets of images, you need to score how similar they are, ranging from 0-10. The number of images could be different in image sets.\n"
    prompt += '[IMG_SET_1]:'

    # Extract matching pairs from text evaluation feedback
    # EXACT MATCH to Reference ref_eval.py lines 235-236
    ref_in_pred = re.findall(r'\(.*?\)', evid_val_score['detailed_ref_in_pred'])
    pred_in_ref = re.findall(r'\(.*?\)', evid_val_score['detailed_pred_in_ref'])

    print('ref in pred:', ref_in_pred, '\n pred in ref', pred_in_ref)

    image_scores = {'pred_in_ref': [], 'ref_in_pred': []}

    # Process pred_in_ref pairs - EXACT MATCH to Reference ref_eval.py lines 241-312
    for detail in pred_in_ref:
        info = detail[1:-1].split(',')
        try:
            pred_idx = int(info[0].split('_')[-1])
            ref_idx = int(info[1].split('_')[-1])
            imgs_pred = pred_evid[pred_idx-1]['images']
            imgs_ref = ref_evid[ref_idx-1]['images']
            if len(imgs_pred) == 0 or len(imgs_ref) == 0:
                feedback = '10'
            else:
                if 'gemini' in model_name:
                    inputs = [prompt]
                    for img in imgs_pred:
                        inputs.append(Image.open(img).convert('RGB'))
                    inputs.append('\n[IMG_SET_2]:')
                    for img in imgs_ref:
                        inputs.append(Image.open(img).convert('RGB'))
                    inputs.append('\nPlease generate your rating with one integer:')
                    response = model.models.generate_content(
                        model='gemini-2.0-flash-001',
                        contents=inputs
                    )
                    feedback = response.text
                elif 'gemma' in model_name:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }]
                    for img in imgs_pred:
                        messages[0]["content"].append({'type': 'image', 'image': img})
                    messages[0]["content"].append({'type': 'text', "text": '\n[IMG_SET_2]:'})
                    for img in imgs_ref:
                        messages[0]["content"].append({'type': 'image', 'image': img})
                    messages[0]["content"].append({'type': 'text', "text": '\nPlease generate your rating with one integer:'})

                    inputs = model["processor"].apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                    ).to(model['model'].device)
                    with torch.no_grad():
                        generated_ids = model['model'].generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    feedback = model["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        except:
            print('##Edge case image!!')
            feedback = '10'

        image_scores['pred_in_ref'].append({
            'info': info,
            'score': feedback
        })

    # Process ref_in_pred pairs - EXACT MATCH to Reference ref_eval.py lines 313-367
    for detail in ref_in_pred:
        info = detail[1:-1].split(',')
        try:
            pred_idx = int(info[1].split('_')[-1])
            ref_idx = int(info[0].split('_')[-1])
            imgs_pred = pred_evid[pred_idx-1]['images']
            imgs_ref = ref_evid[ref_idx-1]['images']
            if len(imgs_pred) == 0 or len(imgs_ref) == 0:
                feedback = '10'
            else:
                if 'gemini' in model_name:
                    inputs = [prompt]
                    for img in imgs_pred:
                        inputs.append(Image.open(img).convert('RGB'))
                    inputs.append('\n[IMG_SET_2]:')
                    for img in imgs_ref:
                        inputs.append(Image.open(img).convert('RGB'))
                    inputs.append('\nPlease generate your rating with one integer:')
                    response = model.models.generate_content(
                        model='gemini-2.0-flash-001',
                        contents=inputs
                    )
                    feedback = response.text
                elif 'gemma' in model_name:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }]
                    for img in imgs_pred:
                        messages[0]["content"].append({'type': 'image', 'image': img})
                    messages[0]["content"].append({'type': 'text', "text": '\n[IMG_SET_2]:'})
                    for img in imgs_ref:
                        messages[0]["content"].append({'type': 'image', 'image': img})
                    messages[0]["content"].append({'type': 'text', "text": '\nPlease generate your rating with one integer:'})

                    inputs = model["processor"].apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                    ).to(model['model'].device)
                    with torch.no_grad():
                        generated_ids = model['model'].generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    feedback = model["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        except:
            print('##Edge case image!!')
            feedback = '10'

        image_scores['ref_in_pred'].append({
            'info': info,
            'score': feedback
        })

    return image_scores

# ============================================================================
# Model Loading
# ============================================================================

def load_evaluation_model(model_name, cache_dir=None, api_key=None):
    """Load the evaluation model (Gemini API or Gemma-3 local)"""
    print(f"Loading evaluation model: {model_name}")

    # Check if using Gemini API
    if 'gemini' in model_name:
        from google import genai
        from google.genai.types import HttpOptions

        if not api_key:
            raise ValueError("Gemini API key is required. Use --gemini_api_key argument or set GEMINI_API_KEY environment variable")

        model = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            api_key=api_key
        )
        print(f"✓ Loaded Gemini API model: {model_name}")
        return model

    # Otherwise use local Gemma model
    elif 'gemma' in model_name:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        print(f"  Loading model from HuggingFace...")
        print(f"  Using device_map='auto' for automatic GPU allocation")
        print(f"  Using torch.bfloat16 for memory efficiency")

        kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16
        }
        if cache_dir:
            kwargs['cache_dir'] = cache_dir

        llm = Gemma3ForConditionalGeneration.from_pretrained(model_name, **kwargs)
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir if cache_dir else None
        )

        model = {
            'model': llm.eval(),
            'processor': processor
        }

        # Print device info
        if hasattr(llm, 'hf_device_map'):
            print(f"✓ Model loaded on devices: {llm.hf_device_map}")
        else:
            print(f"✓ Model loaded with device_map='auto'")
        print(f"✓ Model dtype: {llm.dtype}")

        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}. Use 'gemini' or 'gemma' models.")

# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate using official AVerImaTeC evaluation (standalone)')

    # Input: submission.json (official format)
    parser.add_argument('--submission_path', type=str, required=True,
                        help='Path to submission.json (official AVerImaTeC format)')

    # Required inputs
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to ground truth data')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save evaluation results (not used in parallel mode)')

    # Model configuration
    parser.add_argument('--eval_model', type=str, default='google/gemma-3-27b-it',
                        help='Model to use for evaluation (default: google/gemma-3-27b-it, official competition model)')
    parser.add_argument('--api_model', type=str, default='',
                        help='Use API model instead (e.g., gemini-2.0-flash-001). Requires --gemini_api_key')
    parser.add_argument('--cache_dir', type=str, default='',
                        help='Cache directory for models')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for evaluation (e.g., cuda:0, cpu)')
    parser.add_argument('--gemini_api_key', type=str, default='',
                        help='Gemini API key (required if using --api_model with gemini)')

    # Evaluation options
    parser.add_argument('--justification', action='store_true',
                        help='Enable justification evaluation (disabled by default to save time)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')

    # Parallel processing options (for local model evaluation)
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start index for parallel processing (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index for parallel processing (exclusive)')
    parser.add_argument('--per_sample_output_dir', type=str, default=None,
                        help='Directory to save per-sample results for parallel processing')

    return parser.parse_args()

# ============================================================================
# Main Evaluation Logic
# ============================================================================

def evaluate_sample_submission(sample, gt_sample, mllm, mllm_name, image_dir,
                               eval_justification=False, debug=False):
    """Evaluate a single sample from submission.json format.

    Args:
        sample: Dict from submission.json with keys: id, questions, justification, verdict, evidence
        gt_sample: Ground truth sample with keys: questions, label, justification
        mllm: Evaluation model
        mllm_name: Model name
        image_dir: Path to images
        eval_justification: Whether to evaluate justification
        debug: Debug mode
    """
    # Extract ground truth
    gt_questions_full = gt_sample.get('questions', [])
    gt_questions = [info['question'] for info in gt_questions_full]
    gt_verdict = gt_sample.get('label', '')
    gt_justification = gt_sample.get('justification', '')

    # Convert GT questions to evidence format
    gt_evid = []
    for qa in gt_questions_full:
        evid_info = convert_qa_format(qa, mllm, mllm_name, image_dir)
        gt_evid.append(evid_info)

    # Extract predictions from submission format
    pred_questions = sample.get('questions', [])
    pred_verdict = sample.get('verdict', '')
    pred_justification = sample.get('justification', '')
    pred_evid = sample.get('evidence', [])  # Already in {'text': ..., 'images': [...]} format

    # 1. Verdict evaluation (exact match, case-insensitive)
    verdict_acc = 1.0 if pred_verdict.lower().strip() == gt_verdict.lower() else 0.0

    # 2. Evidence evaluation
    detailed_evid_val, evid_val_score = val_evid_idv(
        mllm, mllm_name, pred_evid, gt_evid, False, True
    )
    img_scores = compute_image_scores(
        mllm, mllm_name, pred_evid, gt_evid, evid_val_score
    )
    _, evid_acc, _ = get_auto_recall(
        detailed_evid_val, img_scores, len(gt_evid), len(pred_evid)
    )

    # 3. Justification evaluation (optional)
    if eval_justification:
        justi_feedback, justi_score = textual_val_single(
            gt_justification, pred_justification,
            image_dir, mllm_name, mllm,
            'justification', debug
        )
        justi_acc = justi_recall_compute(justi_feedback, justi_score)
    else:
        justi_feedback = "Justification evaluation skipped (use --justification to enable)"
        justi_score = {}
        justi_acc = 0.0

    # 4. Question evaluation
    ques_feedback, ques_score = textual_val_single(
        gt_questions, pred_questions,
        image_dir, mllm_name, mllm,
        'question', debug
    )
    ques_acc = ques_recall_compute(
        ques_score, len(gt_questions), len(pred_questions)
    )

    # Debug output
    if debug:
        print('##Question:\n', ques_feedback, '\n', ques_score, '\n\t', ques_acc)
        print('##Verdict:\n', pred_verdict, gt_verdict, verdict_acc)
        print('##Evidence:\n', detailed_evid_val, '\n', img_scores, '\n\t', evid_acc)
        print('##Justification:\n', justi_feedback, '\n', justi_score, '\n\t', justi_acc)

    return {
        'ques_score': ques_acc,
        'evid_score': evid_acc,
        'verdict_score': verdict_acc,
        'justi_score': justi_acc,
        'intermediate_info': {
            'pred_questions': pred_questions,
            'gt_questions': gt_questions,
            'pred_verdict': pred_verdict,
            'gt_verdict': gt_verdict,
            'pred_justification': pred_justification[:500] if pred_justification else '',
            'gt_justification': gt_justification[:500] if gt_justification else '',
            'ques_feedback': ques_feedback,
            'evid_feedback': detailed_evid_val,
            'num_pred_evidence': len(pred_evid),
            'num_gt_evidence': len(gt_evid)
        }
    }


def main():
    args = parse_args()

    # Determine which model to use
    if args.api_model:
        mllm_name = args.api_model
        api_key = args.gemini_api_key
        if not api_key and 'gemini' in mllm_name:
            api_key = os.environ.get('GEMINI_API_KEY', '')
            if not api_key:
                print("\n⚠️  ERROR: No Gemini API key provided!")
                print("   Set GEMINI_API_KEY environment variable or use --gemini_api_key argument\n")
                return
    else:
        mllm_name = args.eval_model
        api_key = None

    # Load evaluation model
    mllm = load_evaluation_model(mllm_name, args.cache_dir, api_key)

    # Load data
    print("Loading data...")
    with open(args.ground_truth_path, 'r') as f:
        gt_data_list = json.load(f)
    gt_data = {i: sample for i, sample in enumerate(gt_data_list)}

    # Load submission.json (official format)
    print(f"\n📋 Loading submission: {args.submission_path}")
    with open(args.submission_path, 'r') as f:
        submission_data = json.load(f)

    if isinstance(submission_data, list):
        submission_dict = {item['id']: item for item in submission_data}
    else:
        submission_dict = submission_data

    all_samples = list(submission_dict.values())
    total_samples = len(all_samples)
    print(f"  → Loaded {total_samples} samples")

    # Apply start_idx and end_idx for parallel processing
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total_samples

    # Clamp indices
    start_idx = max(0, min(start_idx, total_samples))
    end_idx = max(start_idx, min(end_idx, total_samples))

    samples = all_samples[start_idx:end_idx]

    # Check if parallel mode (per-sample output)
    parallel_mode = args.per_sample_output_dir is not None

    if parallel_mode:
        print(f"\n🔄 Parallel processing mode: samples [{start_idx}:{end_idx}]")
        print(f"  Per-sample outputs: {args.per_sample_output_dir}/")
        os.makedirs(args.per_sample_output_dir, exist_ok=True)

    print(f"\nEvaluating {len(samples)} samples (of {total_samples} total)...")
    if args.justification:
        print("  ✓ Justification evaluation: ENABLED")
    else:
        print("  ✗ Justification evaluation: DISABLED (use --justification to enable)")

    # Evaluate each sample
    all_eval_results = []
    import time
    start_time = time.time()

    for i, sample in enumerate(samples):
        if args.debug and i > 4:
            break

        sample_start = time.time()
        sample_id = sample['id']

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] Evaluating sample ID: {sample_id}")
        print(f"{'='*70}")

        gt_sample = gt_data[sample_id]
        num_gt_questions = len(gt_sample.get('questions', []))
        num_pred_questions = len(sample.get('questions', []))
        num_evidence = len(sample.get('evidence', []))

        print(f"  → GT questions: {num_gt_questions}, Pred questions: {num_pred_questions}")
        print(f"  → Evidence items: {num_evidence}")

        # Evaluate using submission format
        try:
            scores = evaluate_sample_submission(
                sample, gt_sample, mllm, mllm_name,
                args.image_dir, args.justification, args.debug
            )
        except Exception as e:
            print(f"\n  ✗ Error evaluating sample: {e}")
            import traceback
            traceback.print_exc()
            scores = {
                'ques_score': 0.0,
                'evid_score': 0.0,
                'verdict_score': 0.0,
                'justi_score': 0.0,
                'intermediate_info': {'error': str(e)}
            }

        # Add sample_id to scores for merging later
        scores['sample_id'] = sample_id
        all_eval_results.append(scores)

        # Save per-sample result in parallel mode
        if parallel_mode:
            per_sample_path = os.path.join(args.per_sample_output_dir, f'{sample_id}.json')
            with open(per_sample_path, 'w') as f:
                json.dump(scores, f, indent=2)

        # Progress summary
        sample_time = time.time() - sample_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = len(samples) - (i + 1)
        if args.debug:
            remaining = min(5 - (i + 1), remaining)
        eta = avg_time * remaining

        print(f"\n  ✓ Completed in {sample_time:.1f}s")
        print(f"  → Scores: Q={scores['ques_score']:.3f} E={scores['evid_score']:.3f} V={scores['verdict_score']:.3f} J={scores['justi_score']:.3f}")
        print(f"  → Progress: {i+1}/{len(samples)} ({(i+1)/len(samples)*100:.1f}%)")
        print(f"  → Avg: {avg_time:.1f}s/sample | ETA: {eta/60:.1f} min")
        if parallel_mode:
            print(f"  → Saved: {per_sample_path}")

    # Compute aggregate scores
    total_time = time.time() - start_time

    print(f"\n\n{'='*70}")
    print(f"EVALUATION COMPLETED!")
    print(f"{'='*70}")

    ques_scores = [r['ques_score'] for r in all_eval_results]
    evid_scores = [r['evid_score'] for r in all_eval_results]
    verdict_scores = [r['verdict_score'] for r in all_eval_results]
    justi_scores = [r['justi_score'] for r in all_eval_results]

    # Compute conditional evidence-verdict and evidence-justification scores
    # Based on Reference/AVerImaTec_Shared_Task/prepare_submission/ipython/Eval_Score_Compute.ipynb
    EVIDENCE_THRESHOLD = 0.3  # λ = 0.3 as per competition rules

    evidence_verdict_scores = []
    evidence_justification_scores = []
    for r in all_eval_results:
        if r['evid_score'] > EVIDENCE_THRESHOLD:
            evidence_verdict_scores.append(r['verdict_score'])
            evidence_justification_scores.append(r['justi_score'])
        else:
            evidence_verdict_scores.append(0.0)
            evidence_justification_scores.append(0.0)

    # Count samples where evidence score exceeds threshold
    num_above_threshold = sum(1 for r in all_eval_results if r['evid_score'] > EVIDENCE_THRESHOLD)

    # Save results
    results = {
        'component_scores': {
            'question_generation': float(np.mean(ques_scores)),
            'evidence_retrieval': float(np.mean(evid_scores)),
            'verdict_prediction': float(np.mean(verdict_scores)),
            'justification': float(np.mean(justi_scores)),
            'evidence_verdict': float(np.mean(evidence_verdict_scores)),  # NEW: Conditional verdict score
            'evidence_justification': float(np.mean(evidence_justification_scores))  # NEW: Conditional justification score
        },
        'num_samples': len(all_eval_results),
        'num_samples_above_evidence_threshold': num_above_threshold,
        'evidence_threshold': EVIDENCE_THRESHOLD,
        'evaluation_time_seconds': total_time,
        'per_sample_results': all_eval_results,
        'note': f'evidence_verdict and evidence_justification scores only count when evidence score > {EVIDENCE_THRESHOLD}'
    }

    # In parallel mode, skip saving aggregate file (use merge script instead)
    if not parallel_mode:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("OFFICIAL EVALUATION RESULTS")
    print("="*70)
    print(f"\nComponent Scores (0-1 scale):")
    print(f"  Question Generation Score:     {results['component_scores']['question_generation']:.4f}")
    print(f"  Evidence Retrieval Score:      {results['component_scores']['evidence_retrieval']:.4f}")
    print(f"  Verdict Prediction Score:      {results['component_scores']['verdict_prediction']:.4f}")
    print(f"  Justification Score:           {results['component_scores']['justification']:.4f}")
    print(f"  Evidence-Verdict Score:        {results['component_scores']['evidence_verdict']:.4f} (conditional)")
    print(f"  Evidence-Justification Score:  {results['component_scores']['evidence_justification']:.4f} (conditional)")
    print(f"\nEvaluation Details:")
    print(f"  Samples evaluated:          {len(all_eval_results)}")
    print(f"  Samples with E > {EVIDENCE_THRESHOLD}:      {num_above_threshold} ({num_above_threshold/len(all_eval_results)*100:.1f}%)")
    print(f"  Total time:                 {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"  Average time per sample:    {total_time/len(all_eval_results):.1f} seconds")
    print(f"\nNote: Evidence-Verdict and Evidence-Justification scores only count when E > {EVIDENCE_THRESHOLD}")
    if parallel_mode:
        print(f"\nPer-sample results saved to: {args.per_sample_output_dir}/")
        print("Run merge script to create final eval_results_local.json")
    else:
        print(f"\nResults saved to: {args.output_path}")
    print("="*70)

if __name__ == '__main__':
    main()


