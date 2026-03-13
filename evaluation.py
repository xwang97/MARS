from pipelines import PipelineRunner
from utils import extract_answer, extract_pred_answer, save_jsonl, load_data, extract_debate_answer, extract_pred_answer_majority
from utils import is_correct, most_frequent_element
from tqdm import tqdm
import numpy as np
from datetime import date
import random
import time
import os


def eval_mars(task="gsm", model=None, n_problems=5, n_reviewers=2, max_rounds=2, selected=True, voting=False, verbosity=0):
    """
    Evaluate the MARVEL framework on certain task using the updated Multi-Round Pipeline.
    """
    all_questions = load_data(task=task)
    mars_scores = []  # scores after multi-agent review
    mars_scores_vote = []
    token_usages = [] 
    records = []  
    rectified_collections = [] 

    # --- Setup Question List (Unchanged) ---
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
        if not os.path.exists(f"data/{task}"):
            os.makedirs(f"data/{task}")
        np.savetxt(f"data/{task}/question_ids.txt", question_list)

    start_time = time.time()
    
    # --- Main Evaluation Loop ---
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
            
        answer = extract_answer(gt_answer, task) # Ground Truth

        # 1. Run the New Multi-Round Pipeline
        runner = PipelineRunner(task=task, model=model)
        # Added max_rounds parameter (you can tune this)
        review_history = runner.run_mars_pipeline(
            question, 
            n_reviewers=n_reviewers, 
            max_rounds=max_rounds, 
            verbosity=verbosity
        )

        # 2. Extract Single Agent Answer (Baseline)
        # The 'input_answer' to Round 1 is the Author's initial, unreviewed draft.
        initial_draft = review_history['rounds'][0]['input_answer']
        single_agent_answer = extract_pred_answer(initial_draft, task)

        # 3. Extract MARS Final Answer
        # The pipeline now explicitly returns the final accepted content
        mars_final_content = review_history['final_answer']
        mars_answer = extract_pred_answer(mars_final_content, task)

        # 4. Extract Voting Answer (Optional Adapter)
        # Your old `extract_pred_answer_majority` likely expects keys like "review1", "review2".
        # We need to convert the *last round's* reviews to that format.
        mars_answer_vote = None
        if voting:
            # 1. Create a dictionary to mimic the old flat structure
            voting_dict = {}

            # A. Add Reviewers (from the LAST round)
            last_round = review_history['rounds'][-1]
            last_round_reviews = last_round['reviews']
            
            for idx, rev_item in enumerate(last_round_reviews):
                # rev_item is { "reviewer_X": "content" }
                content = list(rev_item.values())[0] 
                voting_dict[f"review{idx+1}"] = content

            # B. Add Author's Initial Answer (Round 0 input)
            # Old function needs: 'author_response'
            voting_dict['author_response'] = review_history['rounds'][0]['input_answer']

            # C. Add Author's Final/Rebuttal Answer
            # Old function needs: 'author_rebuttal'
            voting_dict['author_rebuttal'] = review_history['final_answer']

            # D. Add Meta-Reviewer (from the LAST round)
            # Old function needs: 'meta_review'
            voting_dict['meta_review'] = last_round['meta_review']
            
            # 2. Now pass this 'fake' old-style history to the existing voting function
            mars_answer_vote = extract_pred_answer_majority(voting_dict, n_reviewers, task)
        else:
            # Fallback if voting is off, use the meta-reviewer's result
            mars_answer_vote = mars_answer 

        # --- Scoring Logic (Unchanged) ---
        if is_correct(mars_answer, answer, task):
            mars_scores.append(1)
        else:
            mars_scores.append(0)
            
        if is_correct(mars_answer_vote, answer, task):
            mars_scores_vote.append(1)
        else:
            mars_scores_vote.append(0)

        if verbosity:
            print("GT, single-agent, and multi-agent answer: ", answer, single_agent_answer, mars_answer)
            print("\n")

        # Check for rectification (Initially Wrong -> Finally Correct)
        if not is_correct(single_agent_answer, answer, task) and is_correct(mars_answer, answer, task):
            rectified_collections.append(i)

        # Update record with metadata
        review_history['id'] = int(i)
        review_history['multi_score'] = int(mars_scores[-1])
        review_history['question'] = question
        review_history['gt_answer'] = gt_answer
        
        # 'total_tokens' is already in the new review_history structure
        token_usages.append(review_history['total_tokens'])
        records.append(review_history)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)

    # Save Records
    date_str = date.today().isoformat()
    save_name = f"data/{task}/records/{date_str}.jsonl"
    if not os.path.exists(f"data/{task}/records"):
        os.makedirs(f"data/{task}/records")
    save_jsonl(records, save_name)

    return sum(mars_scores), sum(mars_scores_vote), rectified_collections, np.mean(token_usages), avg_time


def eval_single_agent(task="gsm", model=None, n_problems=5, selected=True, verbosity=0):
    all_questions = load_data(task=task)
    scores = []
    token_usages = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    if selected:
        question_list = list(np.loadtxt(f"data/{task}/question_ids.txt").astype(int))
    else:
        question_list = sorted(random.sample(range(len(all_questions)), n_problems))
        # question_list = range(len(all_questions))
        np.savetxt(f"data/{task}/question_ids.txt", question_list)
    start_time = time.time()
    for i in tqdm(question_list):
        question = all_questions[i]["question"]
        gt_answer = all_questions[i]["answer"]
        if verbosity:
            print("question: ", question)
            print("gt_answer: ", gt_answer)
            print("===================")
        answer = extract_answer(gt_answer, task)
        # Run single-agent pipeline
        runner = PipelineRunner(task=task, model=model)
        agent_history = runner.run_single_agent_pipeline(question, verbosity=verbosity)
        pred_answer = extract_pred_answer(agent_history['response'], task)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
            print("\n")
        if pred_answer is not None and is_correct(pred_answer, answer, task):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        agent_history['id'] = int(i)
        agent_history['score'] = scores[-1]
        agent_history['question'] = question
        agent_history['gt_answer'] = gt_answer
        token_usages.append(agent_history['total_tokens'])
        records.append(agent_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/single_agent_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_self_reflection(task="gsm", model=None, n_problems=5, selected=True, verbosity=0):
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
        runner = PipelineRunner(task=task, model=model)
        reflection_history = runner.run_self_reflection_pipeline(question, verbosity=verbosity)
        initial_answer = extract_pred_answer(reflection_history['response'], task)
        pred_answer = extract_pred_answer(reflection_history['reflection'], task)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
        if pred_answer is not None and pred_answer == answer:
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


def eval_self_consistency(task="gsm", model=None, n_problems=5, n_samples=3, selected=True, verbosity=0):
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
        # Run self-consistency pipeline
        runner = PipelineRunner(task=task, model=model)
        consistency_history = runner.run_self_consistency_pipeline(question, num_samples=n_samples, verbosity=verbosity)
        sampled_answers = [extract_pred_answer(response, task) for response in consistency_history['responses']]
        pred_answer = most_frequent_element(sampled_answers)
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
            print("\n")
        if pred_answer is not None and is_correct(pred_answer, answer, task):
            scores.append(int(1))
        else:
            scores.append(int(0))
            hard_collections.append(i)
        consistency_history['id'] = int(i)
        consistency_history['score'] = int(scores[-1])
        consistency_history['question'] = question
        consistency_history['gt_answer'] = gt_answer
        token_usages.append(consistency_history['total_tokens'])
        records.append(consistency_history)
    end_time = time.time()
    avg_time = (end_time - start_time) / len(question_list)
    date_str = date.today().isoformat()
    save_name = f"baselines/consistency_logs/{task}_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, np.mean(token_usages), avg_time


def eval_debate(task="gsm", model=None, n_problems=5, n_agents=3, selected=True, verbosity=0):
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
        runner = PipelineRunner(task=task, model=model)
        debate_history, total_tokens = runner.run_debate_pipeline(question, num_agents=n_agents, verbosity=verbosity)
        pred_answer = extract_debate_answer(debate_history, task)  # need update !!!
        if verbosity:
            print("GT answer and predicted answer: ", answer, pred_answer)
        if pred_answer is not None and is_correct(pred_answer, answer, task=task):
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