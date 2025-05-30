from agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent
from utils import extract_math_decision, parse_simple_math_answer, extract_pred_answer


def run_detection_pipeline(sentence: str, concept: str):
    # reviewers = create_reviewer_agents(agent_type="toolcall")
    # meta = create_meta_reviewer_agent(agent_type="toolcall")
    reviewers = create_reviewer_agents()
    meta = create_meta_reviewer_agent()

    print("\n[📥 Sentence for Evaluation]:")
    print(sentence)
    print("\n[📘 Context concept]:")
    print(concept)

    # Step 1: Reviewers analyze the sentence with passage context
    review_responses = []
    for reviewer in reviewers:
        review_input = (
            "You are a hallucination reviewer.\n"
            "Your task is to determine whether the following sentence about the given concept is factual.\n"
            "📘 Concept:\n"
            f"{concept}\n\n"
            "📥 Sentence:\n"
            f"{sentence}\n\n"
            "You can only answer factual if it meets all of these criteria: \n"
            "1. Consistent with known facts (e.g., time, places, and other well-known facts on the web).\n"
            "2. Related to the given concept.\n\n"
            "Always keep these things in mind: \n"
            "1. Remember to start by using web_search tool to search on the internet.\n"
            "2. Your output must follow this structure exactly:\n\n"
            "Decision: [factual | non-factual]\n"
            "Confidence: [1-5]\n"
            "Reasons:\n"
            "- Reason 1\n"
            "- Reason 2 (optional)\n"
            "- Reason 3 (optional)\n\n"
            "Remeber the decision must be non-factual if any one of the criteria is violated. Now make your decision and explain your reasoning following the structure above."
        )

        review = reviewer.run(review_input)
        review_responses.append(review)
        print(f"\n--- {reviewer.name} Review ---\n{review}")

    # Step 2: Meta-reviewer synthesizes and decides
    combined_reviews = "\n\n".join(
        [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
    )
    meta_input = (
        "You are a meta-reviewer. Your task is to determine whether the following sentence about the given concept is hallucinated.\n"
        "📘 Concept:\n"
        f"{concept}\n\n"
        "📥 Sentence:\n"
        f"{sentence}\n\n"
        "Your conclusion should rely on the following sources:\n"
        "1. The reviews from multiple agents (which include their decision, confidence, and reasoning)\n"
        "2. Your own general world knowledge and critical judgment\n\n"
        "Important things to keep in mind: \n"
        "Some reviewers may use web search to support their reasoning. "
        "You should trust well-sourced search-backed justifications over unsupported guesses or vague reasoning. "
        "If reviewers cite external evidence, weigh that highly in your final judgment.\n"
        "Each review includes:\n"
        "- A decision: factual or non-factual\n"
        "- A confidence score (1-5)\n"
        "- 1–3 reasons\n\n"
        "Your output must follow this structure exactly:\n\n"
        "Decision: [factual | non-factual]\n"
        "Confidence: [1–5]\n"
        "Justification:\n"
        "- Reason 1\n"
        "- Reason 2 (optional)\n"
        "- Reason 3 (optional)\n\n"
        "--- Reviewer Feedback ---\n"
        f"{combined_reviews}\n\n"
        "📢 Provide your final classification using the structure above. Be concise, clear, and well-justified."
    )

    final_decision = meta.run(meta_input)
    print("\n=== 🧠 Meta-Reviewer Final Judgment ===\n", final_decision)
    return final_decision


def run_simple_math_pipeline(user_query: str):
    author = create_author_agent()
    reviewers = create_reviewer_agents()
    meta = create_meta_reviewer_agent()

    # Step 1: Author answers
    author_input = (
        f"What is the result of {user_query}? Make sure to state your answer at the end of the response."
    )
    author_response = author.run(author_input)
    print("\n=== Author's Answer ===\n", author_response)

    # Step 2: Reviewers critique
    review_responses = []
    for reviewer in reviewers:
        review_input = (
            "You are a reviewer. The author answered this question:\n\n"
            f"Question: {user_query}\n\n"
            f"Answer: {author_response}\n\n"
            "Review the answer for potential inaccuracies. Give your conclusion and comments with the following format:\n\n"
            "Decision: [right | wrong]\n"
            "Confidence: [1–5]\n"
            "Justification:\n"
            "- Reason 1\n"
            "- Reason 2 (optional)\n"
            "- Reason 3 (optional)\n\n"
        )
        review = reviewer.run(review_input)
        review_responses.append(review)
        print(f"\n--- {reviewer.name} Review ---\n{review}")

    # Step 3: Meta-review
    combined_reviews = "\n\n".join(
        [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
    )
    meta_input = (
        "You are the meta-reviewer. The author answered this question:\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: {author_response}\n\n"
        "You should decide whether the answer is correct based on both you own knowledge and the reviewers' comments."
        "--- Reviewer Feedback ---\n"
        f"{combined_reviews}\n\n"
        "Give you conclusion and comments with the following format:\n\n"
        "Decision: [right | wrong]\n"
        "Justification:\n"
        "- Reason 1\n"
        "- Reason 2 (optional)\n"
        "- Reason 3 (optional)\n\n"
    )
    meta_decision = meta.run(meta_input)
    print("\n=== Meta-Reviewer Final Decision ===\n", meta_decision)

    # Step 4: Send feedback or return final answer
    decision = extract_simple_math_decision(meta_decision)
    answer = parse_simple_math_answer(f"{author_response}")
    if decision == "wrong" or decision == "Wrong":
        feedback_input = (
            f"Your answer of {user_query} is {answer}. However, the meta-reviewer said it's wrong.\n\n"
            "Please look at the following comments and try to update you answer.\n\n"
            f"{meta_decision}\n\n."
            "Make sure to state your new answer at the end."
        )
        author_rebuttal = author.run(feedback_input)
        print("\n=== Author's new answer ===\n", author_rebuttal)
        updated_answer = parse_simple_math_answer(author_rebuttal)
        return updated_answer
    return answer


def run_gsm_pipeline(user_query):
    author = create_author_agent()
    reviewers = create_reviewer_agents()
    meta = create_meta_reviewer_agent()

    # Step 1: Author answers
    author_input = (
        "You are a math assistant. Please help to solve the following math problem:\n"
        f"{user_query}\n\n"
        "Give your computation steps in the following format:\n"
        "Thoughts: [your step-by-step computation process]\n"
        "Your final answer should be a single numerical number, in the form"
        "Answer: [the final numerical answer], at the end of your response.\n\n"
    )
    author_response = author.run(author_input)
    print("\n=== Author's Answer ===\n", author_response)

    # Step 2: Reviewers critique
    review_responses = []
    for reviewer in reviewers:
        review_input = (
            "You are a reviewer. The author answered this question:\n\n"
            f"Question: {user_query}\n\n"
            f"Answer: {author_response}\n\n"
            "Review the answer for potential inaccuracies. Give your decision and comments with the following format:\n\n"
            "Decision: [right | wrong]\n"
            "Confidence: [1–5]\n"
            "Justification:\n"
            "- Reason 1\n"
            "- Reason 2 (optional)\n"
            "- Reason 3 (optional)\n"
            "If the decision is wrong, your justification should point out mistakes in the author's thoughts.\n\n"
        )
        review = reviewer.run(review_input)
        review_responses.append(review)
        print(f"\n--- {reviewer.name} Review ---\n{review}")

    # Step 3: Meta-review
    combined_reviews = "\n\n".join(
        [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
    )
    meta_input = (
        "You are the meta-reviewer. The author answered this question:\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: {author_response}\n\n"
        "You should decide whether the answer is correct based on both you own knowledge and the reviewers' comments.\n"
        "--- Reviewer Feedback ---\n"
        f"{combined_reviews}\n\n"
        "Give you conclusion and comments with the following format."
        "If the decision is wrong, you must tell which step is wrong and give your suggestions.\n\n"
        "Decision: [right | wrong]\n"
        "Justification:\n"
        "- Reason 1\n"
        "- Reason 2 (optional)\n"
        "- Reason 3 (optional)\n"
        "Suggestions: [your suggestions for updating the answer]\n\n"
    )
    meta_decision = meta.run(meta_input)
    print("\n=== Meta-Reviewer Final Decision ===\n", meta_decision)
    print("\n")

    # Additional step: build a dictionary to save the review process
    review_history = {
        "author_response": author_response,
        "review1": review_responses[0], "review2": review_responses[1], "review3": review_responses[2],
        "meta_review": meta_decision
    }

    # Step 4: Send feedback or return final answer
    decision = extract_math_decision(meta_decision)
    author_answer = extract_pred_answer(author_response)
    if decision == "wrong" or decision == "Wrong":
        feedback_input = (
            f"Your answer of {user_query} is {author_answer}. However, the meta-reviewer said it's wrong.\n\n"
            "You must consider the meta-reviewer's suggested answer seriously and try to update you answer.\n\n"
            f"{meta_decision}\n\n."
            "Make sure to state your thoughts and new answer with this format:\n"
            "Thoughts: [your step-by-step computation process]\n"
            "Answer: [the final numerical answer]\n"
            "Your final answer should be a single numerical number at the end of your response.\n\n"
        )
        author_rebuttal = author.run(feedback_input)
        print("\n=== Author's new answer ===\n", author_rebuttal)
        review_history['author_rebuttal'] = author_rebuttal
    return review_history