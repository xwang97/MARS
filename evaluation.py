from workflow import run_detection_pipeline, run_simple_math_pipeline, run_gsm_pipeline
from utils import extract_decision_label, read_jsonl, extract_answer, extract_pred_answer, save_jsonl
from tqdm import tqdm
import datasets
import numpy as np
from datetime import date


def eval_selfcheck(data):
    """
    Evaluate our framework on the selfcheckGPT dataset.
    Input:
        data: the dataset loaded by the get_selfcheck_data function
    Return:
        true_labels: the ground truth labels of each sentence
        pred_labels: the predicted labels of each sentence
    """
    wikibio = datasets.load_dataset("wiki_bio", split="test")
    true_labels = []  # true labels of each sentence
    pred_labels = []  # predicted labels of each sentence
    for i in tqdm(range(len(data))):
        if i == 1: break
        row = data[i]
        wiki_id = row["wiki_bio_test_idx"]
        concept = wikibio[wiki_id]["input_text"]["context"].strip()
        passage = row["gpt3_text"]
        for j, sentence in enumerate(row["gpt3_sentences"]):
            print(j)
            # define the prompt, response and labels as per the dataset
            # prompt = f"This is a Wikipedia passage about {concept}:"
            if row["annotation"][j] == "major_inaccurate":
                label = 1
            elif row["annotation"][j] == "minor_inaccurate":
                label = 0.5
            elif row["annotation"][j] == "accurate":
                label = 0
            else:
                raise ValueError("Invalid annotation")
            true_labels.append(label)
            # run the detection pipeline
            final_decision = run_detection_pipeline(sentence, concept)
            pred = extract_decision_label(final_decision)
            pred_labels.append(pred)
    return true_labels, pred_labels


def eval_simple_math(n_problems=10):
    """
    Generate random simple math problems and evaluate the performance.
    """
    np.random.seed(0)
    scores = []
    for _ in tqdm(range(n_problems)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)
        answer = a + b * c + d - e * f
        query = f"{a}+{b}*{c}+{d}-{e}*{f}"
        print(query, answer)
        author_answer = run_simple_math_pipeline(query)
        if float(author_answer) == answer:
            scores.append(1)
        else:
            scores.append(0)
    return sum(scores)


def eval_gsm(n_problems=5, selected=False):
    """
    Randomly fetch n_problems GSM samples and evaluate.
    """
    gsm = read_jsonl('data/GSM/test.jsonl')
    scores = []
    records = []  # record the full review process of each question
    hard_collections = []  # record the incorrectly answered questions
    rectified_collections = []  # record initially wrong but rectified by reviewers questions
    if selected:
        question_list = list(np.loadtxt("data/GSM/hard.txt").astype(int))
    else:
        question_list = list(range(n_problems))
    for i in tqdm(question_list):
        question = gsm[i]["question"]
        gt_answer = gsm[i]["answer"]
        print("question: ", question)
        print("gt_answer: ", gt_answer)
        print("===================")
        answer = extract_answer(gt_answer)
        review_history = run_gsm_pipeline(question)
        if "author_rebuttal" not in review_history:  # initial answer accepted
            pred_answer = extract_pred_answer(review_history['author_response'])
            print("GT answer and predicted answer: ", answer, pred_answer)
            print("\n")
            if float(pred_answer) == float(answer):
                scores.append(1)
            else:
                scores.append(0)
                hard_collections.append(i)
        else:
            initial_answer = extract_pred_answer(review_history['author_response'])
            updated_answer = extract_pred_answer(review_history['author_rebuttal'])
            print("GT answer, initial answer, final answer: ", answer, initial_answer, updated_answer)
            print("\n")
            if float(updated_answer) == float(answer):
                scores.append(1)
                if float(initial_answer) != float(answer):
                    rectified_collections.append(i)
                    hard_collections.append(i)
            else:
                scores.append(0)
                hard_collections.append(i)
        review_history['id'] = i
        review_history['score'] = scores[-1]
        records.append(review_history)
    # save all the review histories
    date_str = date.today().isoformat()
    save_name = f"data/GSM/record_{date_str}.jsonl"
    save_jsonl(records, save_name)
    return sum(scores), hard_collections, rectified_collections

# !!! Update plan
# 1. Write a reviewer tool to check whether the author's thought is consistent with the question.
# 2. Often fail when there are large numbers, maybe we can write some tools
# 3. Test other tasks.