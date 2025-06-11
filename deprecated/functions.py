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