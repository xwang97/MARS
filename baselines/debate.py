import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm
from datetime import date
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_agents import create_author_agent
from utils import extract_answer, extract_pred_answer, most_frequent_element, save_jsonl, read_jsonl


def extract_debate_answer(agent_histories):
    final_responses = [history[-1]["content"] for history in agent_histories]
    answers = [extract_pred_answer(r) for r in final_responses]
    majority = most_frequent_element(answers)
    return majority


def construct_debate_prompt(other_agents_responses, user_query, response_idx):
    if not other_agents_responses:
        return {
            "role": "user",
            "content": (
                "You are a math assistant. Please help to solve the following math problem:\n"
                f"{user_query}\n\n"
                "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
                "Thoughts: [your step-by-step computation process with immediate results]\n"
                "Answer: [the final numerical answer]\n\n"
                "Your final answer must be a single numerical number at the end of the response.\n\n"
            )
        }

    prompt = "These are the solutions to the problem from other agents:\n"
    for history in other_agents_responses:
        response = history[response_idx]["content"]
        prompt += f"\n\nOne agent solution: ```{response}```"

    prompt += (
        "\n\nUsing the solutions from other agents as additional information, can you provide your final answer to the math problem?\n"
        "Make sure to state your thoughts and new answer with this format:\n"
        "Thoughts: [your step-by-step computation process]\n"
        "Answer: [the final numerical answer]\n"
        "Your final answer must be a single numerical number at the end of the response.\n\n"
    )
    return {"role": "user", "content": prompt}


def run_debate_gsm(user_query: str, num_agents: int=3, num_rounds: int=2) -> list[list[dict]]:
    agents = [
        create_author_agent(name=f"Agent_{i+1}")
        for i in range(num_agents)
    ]
    agent_histories = [[] for _ in range(num_agents)]

    # Round 0: each agent answers independently
    for i in range(num_agents):
        prompt = construct_debate_prompt([], user_query, response_idx=0)
        agent_histories[i].append(prompt)
        response = agents[i].run(agent_histories[i])
        agent_histories[i].append(response)

    # Rounds >= 1: agents revise based on others
    for r in range(1, num_rounds):
        for i in range(num_agents):
            other_histories = agent_histories[:i] + agent_histories[i+1:]
            # print(r, i, other_histories)
            prompt = construct_debate_prompt(other_histories, user_query, response_idx=2*r - 1)
            agent_histories[i].append(prompt)
            response = agents[i].run(agent_histories[i])
            agent_histories[i].append(response)

    return agent_histories  # List of message histories per agent


def eval_debate_gsm(n_problems=5, selected=True):
    gsm = read_jsonl('data/GSM/test.jsonl')
    scores = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt("data/GSM/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(gsm)), n_problems))
    for i in tqdm(question_list):
        question = gsm[i]["question"]
        gt_answer = gsm[i]["answer"]
        print("question: ", question)
        print("gt_answer: ", gt_answer)
        print("===================")
        answer = extract_answer(gt_answer)
        debate_history = run_debate_gsm(question)
        pred_answer = extract_debate_answer(debate_history)
        print("GT answer and predicted answer: ", answer, pred_answer)
        if float(pred_answer) == float(answer):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        records.append({"id": int(i), "score": int(scores[-1]), "debate_history": debate_history})
    date_str = date.today().isoformat()
    save_name = f"baselines/debate_logs/gsm_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections