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