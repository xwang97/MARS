import re
import datasets
import json
from collections import Counter
from glob import glob
import pandas as pd



###################################################
# 0. General usage
###################################################
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def save_jsonl(samples, save_name):
    with open(save_name, "w", encoding="utf-8") as f:
        for item in samples:
            json.dump(item, f)
            f.write("\n")


def get_openai_api_key(filename):
    """
    Read OpenAI API key from txt file.
    """
    api_key = ""
    with open(filename) as file:
        api_key = file.read()
    return api_key


def most_frequent_element(lst):
    """
    Return the most frequent element of a list
    """
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]


def load_data(task):
    if task == "gsm":
        all_questions = read_jsonl('data/gsm/test.jsonl')
    if task == "mmlu":
        # Add your code here
        tasks = glob("data/mmlu/test/*.csv")
        dfs = [pd.read_csv(task) for task in tasks]
        # print(len(dfs))
        all_questions=[]
        for df in dfs:
            for i in range(len(df)):
                question = df.iloc[i, 0]
                a = df.iloc[i, 1]
                b = df.iloc[i, 2]
                c = df.iloc[i, 3]
                d = df.iloc[i, 4]
                question = "{}: A) {}, B) {}, C) {}, D) {}".format(question, a, b, c, d)
                answer = df.iloc[i, 5]
                single_que={"question":question,
                        "answer":answer}
                all_questions.append(single_que)
        # print(all_questions[1]["answer"])
    return all_questions


###################################################
# 1. Hallucination detection related
###################################################
def extract_decision_label(text: str | dict) -> int:
    """
    Extract decision from the final output of the detection task. Return 1 for
    non-factual, 0 for factual or bad-formatted answer.
    """
    if isinstance(text, dict):
        decision = text['Decision']
        if decision == "non-factual":
            return 1
    else:
        match = re.search(r'Decision:\s*(\w+[-]?\w*)', text)
        if match:
            decision = match.group(1)
            if decision == "non-factual":
                return 1
    return 0


def get_selfcheck_data(n_samples=1000):
    """
    Load the selfcheckGPT dataset from huggingface hub.
    """
    dataset = datasets.load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    return dataset["evaluation"]


###################################################
# 2. Simple math problems related
###################################################
def parse_simple_math_answer(sentence):
    sentence = str(sentence)
    parts = sentence.split(" ")
    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


###################################################
# 3. GSM related
###################################################
def extract_answer(text, task):
    """
    Extract answer from the given sample.
    """
    if task == "gsm":
        if isinstance(text, int):
            return float(text)
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"
        match = ANS_RE.search(text)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return float(match_str)
        else:
            return float(parse_simple_math_answer(text))
        return INVALID_ANS
    if task == "mmlu":
        return text[-1]


def extract_pred_answer(text, task):
    """
    Extract answer from the model response
    """
    if task == "gsm":
        if not isinstance(text, str):
            text = str(text)
        matches = re.findall(r"\$?\d[\d,]*\.?\d*", text)
        if matches:
            last = matches[-1]
            # Remove $ and commas
            return float(re.sub(r"[^\d.]", "", last))
        return None
    if task == "mmlu":
        # 1. Try direct "Answer: X" line
        match = re.search(r'Answer:\s*([ABCD])[\).]?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # 2. Fallback to "I would say that X)" or similar
        match = re.search(r'I would say that\s+([ABCD])[\).]?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # 3. Optional: fallback to last standalone A)-D) in the text (least reliable)
        match = re.findall(r'\b([ABCD])[\).]', text)
        if match:
            return match[-1].upper()  # last one is likely the final answer
        return None  # no answer found


def extract_pred_answer_majority(review_history, n_reviewers, task):
    """
    Vote for the final answer, where the author has a higher priority
    """
    ans_list = []
    initial_answer = extract_pred_answer(review_history['author_response'], task)
    ans_list.append(initial_answer)
    if 'author_rebuttal' not in review_history:
        ans_list.append(initial_answer)
    else:
        updated_answer = extract_pred_answer(review_history['author_rebuttal'], task)
        ans_list.append(updated_answer)
    for i in range(n_reviewers):
        review = review_history[f"review{i+1}"]
        ans_list.append(extract_pred_answer(review, task))
    return most_frequent_element(ans_list)


def extract_math_decision(text) -> str:
    """
    Extracts decision of the meta-reviewer for a math problem.
    """
    if isinstance(text, dict):
        if "Decision" in text.keys():
            decision = text['Decision']
        elif "decision" in text.keys():
            decision = text['decision']
        else:
            decision = None
    else:
        text = str(text)
        match = re.search(r'Decision:\s*(\w+[-]?\w*)', text)
        if match:
            decision = match.group(1)
        else:
            match = re.search(r"right|wrong|Right|Wrong", text)
            if match:
                return match.group(0)
            return "right"
    if decision is None:
        decision = "right"
    return decision


def extract_debate_answer(agent_histories, task):
    if task == "gsm":
        final_responses = [history[-1]["content"] for history in agent_histories]
        answers = [extract_pred_answer(r, task=task) for r in final_responses]
        majority = most_frequent_element(answers)
    if task == "mmlu":
        final_responses = [history[-1]["content"] for history in agent_histories]
        answers = [extract_pred_answer(r, task=task) for r in final_responses]
        majority = most_frequent_element(answers)
    return majority