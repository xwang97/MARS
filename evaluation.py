from pipelines import PipelineRunner
from utils import extract_answer, extract_pred_answer, save_jsonl, load_data, extract_debate_answer
from tqdm import tqdm
import numpy as np
from datetime import date
import random
import time


def eval_marvel(task="gsm", n_problems=5, n_reviewers=3, selected=False, verbosity=0):
    """
    Evaluate the MARVEL framework on certain task.
    """
    all_questions = load_data(task=task)  # need update !!!
    single_agent_scores = []  # scores of single agent
    multi_agent_scores = []  # scores after multi-agent review
    token_usages = []  # total token consumptions of each question
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    rectified_collections = []  # record initially wrong but rectified by reviewers questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
        np.savetxt(f"data/{task}/question_ids.txt", question_list)
    # start testing on each question
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)  # need update !!!
        # Run MARVEL pipeline
        runner = PipelineRunner(task=task)
        review_history = runner.run_marvel_pipeline(question, n_reviewers=n_reviewers, verbosity=verbosity)
        if "author_rebuttal" not in review_history:  # initial answer accepted
            pred_answer = extract_pred_answer(review_history['author_response'], task)  # need update !!!
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
            initial_answer = extract_pred_answer(review_history['author_response'], task)
            updated_answer = extract_pred_answer(review_history['author_rebuttal'], task)
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
    save_name = f"data/{task}/records/{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(single_agent_scores), sum(multi_agent_scores), hard_collections, rectified_collections, np.mean(token_usages), avg_time


def eval_self_reflection(task="gsm", n_problems=5, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run reflection pipeline
        runner = PipelineRunner(task=task)
        reflection_history = runner.run_self_reflection_pipeline(question, verbosity=verbosity)
        pred_answer = extract_pred_answer(reflection_history['reflection'], task)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
        if float(pred_answer) == float(answer):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        reflection_history['id'] = int(i)
        reflection_history['score'] = scores[-1]
        token_usages.append(reflection_history['total_tokens'])
        records.append(reflection_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/reflection_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_debate(task="gsm", n_problems=5, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run debate pipeline
        runner = PipelineRunner(task=task)
        debate_history, total_tokens = runner.run_debate_pipeline(question, verbosity=verbosity)
        pred_answer = extract_debate_answer(debate_history, task)  # need update !!!
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
        if float(pred_answer) == float(answer):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        token_usages.append(total_tokens)
        records.append({"id": int(i), "score": int(scores[-1]), "total_tokens": total_tokens, "debate_history": debate_history})
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/debate_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time