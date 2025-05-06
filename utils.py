import re
import datasets


def extract_decision_label(text: str) -> int:
    """
    Extract decision from the final output of the detection task. Return 1 for
    non-factual, 0 for factual or bad-formatted answer.
    """
    match = re.search(r'Decision:\s*(\w+[-]?\w*)', text)
    if match:
        decision = match.group(1)
        if decision == "non-factual":
            return 1
        else:
            return 0
    return 0


def get_selfcheck_data(n_samples=1000):
    """
    Load the selfcheckGPT dataset from huggingface hub.
    """
    dataset = datasets.load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    return dataset["evaluation"]