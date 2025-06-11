import sys
import os
import numpy as np
import random
from tqdm import tqdm
from datetime import date
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_agents import create_author_agent
from utils import read_jsonl, extract_answer, extract_pred_answer, save_jsonl


def construct_initial_prompt(user_query, task):
    if task == "gsm":
        initial_prompt = (
            "You are a math assistant. Please help to solve the following math problem:\n"
            f"{user_query}\n\n"
            "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
            "Thoughts: [your step-by-step computation process with immediate results]\n"
            "Answer: [the final numerical answer]\n\n"
            "Your final answer must be a single numerical number at the end of the response.\n\n"
        )
    return initial_prompt

def construct_reflection_prompt(user_query, response, task):
    if task == "gsm":
        reflection_prompt = (
            "You wrote the following response to a math problem:\n\n"
            f"Qustion: {user_query}\n\n"
            f"Answer: {response}\n\n"
            "Carefully review your own answer. Are there any mistakes, inconsistencies, or calculation errors?\n"
            "If yes, explain the problems and revise your answer accordingly. If not, confirm and repeat your initial answer."
            "Your final response must follow this format:\n"
            "Mistakes (if any): \n\n"
            "Answer: [the final single numerical answer]\n\n"
        )
    return reflection_prompt


def run_self_reflection_pipeline(user_query, task, verbosity=0):
    agent = create_author_agent()
    # Step 1: Initial answer
    author_input = construct_initial_prompt(user_query, task)
    response = agent.run(author_input)
    if verbosity:
        print("\n=== Initial Answer ===\n", response)

    # Step 2: Self-reflection
    reflection_prompt = construct_reflection_prompt(user_query, response, task)
    reflection = agent.run(reflection_prompt)
    if verbosity:
        print("\n=== Final answer after self-reflection ===\n", reflection)
    reflection_history = {"response": response, "reflection": reflection, "total_tokens": agent.total_tokens}
    return reflection_history


def eval_self_reflection_gsm(n_problems=5, selected=True, verbosity=0):
    gsm = read_jsonl('data/GSM/test.jsonl')
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt("data/GSM/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(gsm)), n_problems))
    start_time = time.time()
    for i in tqdm(question_list):
        question = gsm[i]["question"]
        gt_answer = gsm[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer)
        reflection_history = run_self_reflection_pipeline(question, task="gsm", verbosity=verbosity)
        pred_answer = extract_pred_answer(reflection_history['reflection'])
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
    save_name = f"baselines/reflection_logs/gsm_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time