from pipelines import run_review_pipeline
from utils import extract_decision_label, read_jsonl, extract_answer, extract_pred_answer, save_jsonl
from tqdm import tqdm
import numpy as np
from datetime import date
import random
import time


def eval_gsm(n_problems=5, n_reviewers=3, selected=False, verbosity=0):
    """
    Randomly fetch n_problems GSM samples and evaluate.
    """
    gsm = read_jsonl('data/GSM/test.jsonl')
    single_agent_scores = []  # scores of single agent
    multi_agent_scores = []  # scores after multi-agent review
    token_usages = []  # total token consumptions of each question
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    rectified_collections = []  # record initially wrong but rectified by reviewers questions
    if selected:
        question_list = list(np.loadtxt("data/GSM/hard.txt").astype(int))
    else:
        # question_list = list(range(n_problems))
        question_list = sorted(random.sample(range(len(gsm)), n_problems))
        np.savetxt("data/GSM/question_ids.txt", question_list)
    start_time = time.time()
    for i in tqdm(question_list):
        question = gsm[i]["question"]
        gt_answer = gsm[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer)
        review_history = run_review_pipeline(question, task="gsm", n_reviewers=n_reviewers, verbosity=verbosity)
        if "author_rebuttal" not in review_history:  # initial answer accepted
            pred_answer = extract_pred_answer(review_history['author_response'])
            if verbosity:
                print("GT answer and predicted answer: ", answer, pred_answer)
                print("\n")
            if float(pred_answer) == float(answer):
                single_agent_scores.append(1)
                multi_agent_scores.append(1)
            else:
                single_agent_scores.append(0)
                multi_agent_scores.append(0)
                hard_collections.append(i)
        else:
            initial_answer = extract_pred_answer(review_history['author_response'])
            updated_answer = extract_pred_answer(review_history['author_rebuttal'])
            if verbosity:
                print("GT answer, initial answer, final answer: ", answer, initial_answer, updated_answer)
                print("\n")
            if float(initial_answer) == float(answer):
                single_agent_scores.append(1)
            else:
                single_agent_scores.append(0)
            if float(updated_answer) == float(answer):
                multi_agent_scores.append(1)
                if float(initial_answer) != float(answer):
                    rectified_collections.append(i)
                    hard_collections.append(i)
            else:
                multi_agent_scores.append(0)
                hard_collections.append(i)
        review_history['id'] = i
        review_history['single_score'] = single_agent_scores[-1]
        review_history['multi_score'] = multi_agent_scores[-1]
        token_usages.append(review_history['total_tokens'])
        records.append(review_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    # save all the review histories
    date_str = date.today().isoformat()
    save_name = f"data/GSM/records/record_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(single_agent_scores), sum(multi_agent_scores), hard_collections, rectified_collections, np.mean(token_usages), avg_time