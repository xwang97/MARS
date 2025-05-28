import re
import datasets
import json


###################################################
# 0. General usage
###################################################
def get_openai_api_key(filename):
    """
    Read OpenAI API key from txt file.
    """
    api_key = ""
    with open(filename) as file:
        api_key = file.read()
    return api_key


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
            match = re.search(r"right|wrong", text)
            if match:
                return match.group(0)
            return "right"
    if decision is None:
        decision = "right"
    return decision


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
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def extract_answer(text):
    """
    Extract answer from the GSM sample.
    """
    if isinstance(text, int):
        return float(text)
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    match = ANS_RE.search(text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return parse_simple_math_answer(text)
    return INVALID_ANS


def extract_pred_answer(text):
    """
    Extract answer from the model response
    """
    if not isinstance(text, str):
        text = str(text)
    matches = re.findall(r"\$?\d[\d,]*\.?\d*", text)
    if matches:
        last = matches[-1]
        # Remove $ and commas
        return re.sub(r"[^\d.]", "", last)
    return None