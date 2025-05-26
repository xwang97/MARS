import re
import datasets
import json


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


def get_openai_api_key(filename):
    """
    Read OpenAI API key from txt file.
    """
    api_key = ""
    with open(filename) as file:
        api_key = file.read()
    return api_key


def extract_simple_math_decision(text) -> str:
    """
    Extracts decision of the meta-reviewer for a simple math problem.
    """
    if isinstance(text, dict):
        if "Decision" in text.keys():
            decision = text['Decision']
        elif "decision" in text.keys():
            decision = text['decision']
        else:
            decision = None
    else:
        match = re.search(r'Decision:\s*(\w+[-]?\w*)', str(text))
        if match:
            decision = match.group(1)
        else:
            match = re.search(r"right|wrong", text)
            if match:
                return match.group(0)
            return "right"
    if decision is None:
        decision = "right"
    return decision


def parse_simple_math_answer(sentence):
    sentence = str(sentence)
    parts = sentence.split(" ")
    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def extract_answer(gt):
    """
    Extract answer from the GSM sample.
    """
    if isinstance(gt, int):
        return float(gt)
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    match = ANS_RE.search(gt)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    """
    Check whether model completion of GSM question is correct.
    """
    gt_answer = extract_answer(gt_example)
    pred_answer = parse_simple_math_answer(model_completion)
    print("ground truth: ", gt_answer, "pred: ", pred_answer)
    return float(pred_answer) == float(gt_answer)