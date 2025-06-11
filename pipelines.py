from custom_agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent
from utils import extract_math_decision, parse_simple_math_answer, extract_pred_answer


def construct_author_prompt(user_query, task):
    if task == "gsm":
        author_prompt = (
            "You are a math assistant. Please help to solve the following math problem:\n"
            f"{user_query}\n\n"
            "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
            "Thoughts: [your step-by-step computation process with immediate results]\n"
            "Answer: [the final numerical answer]\n\n"
            "Your final answer must be a single numerical number at the end of the response.\n\n"
        )
    return author_prompt


def construct_reviewer_prompt(user_query, author_response, task):
    output_format = (
        "---\n\n"
        "Your output format must be:\n\n"
        "Decision: [right | wrong]  \n"
        "Confidence: [1–5] (5 = highest confidence)  \n"
        "Justification: [reasons or author mistakes supporting your decision] \n"
        "---\n\n"
    )
    if task == "gsm":
        reviewer_prompt = (
            "You are a reviewer. The author has submitted the following answer to a math problem:\n\n"
            f"Question: {user_query}\n\n"
            f"Answer: {author_response}\n\n"
            "Please evaluate the correctness of the author's response. Follow the instructions and format strictly:\n\n"
            f"{output_format}"
        )
    return reviewer_prompt


def construct_meta_prompt(user_query, author_response, combined_reviews, task):
    output_format = (
        "Decision: [right | wrong]\n"
        "Justification: [reasons of your decision]\n"
        "Suggestions: [your suggestions for updating the answer]\n"
    )
    if task == "gsm":
        meta_prompt = (
            "You are the meta-reviewer. The author has submitted an answer to a math problem.\n\n"
            f"Question: {user_query}\n\n"
            f"Answer: {author_response}\n\n"
            "You must decide whether the answer is correct by summarizing and analyzing the reviewers' comments below:\n\n"
            "--- Reviewer Feedback ---\n"
            f"{combined_reviews}\n\n"
            "Provide your conclusion in the following format. If the decision is 'wrong', you must identify the flawed step(s) and give your suggestions for revision.\n\n"
            f"{output_format}"
        )
    return meta_prompt


def construct_feedback_prompt(user_query, author_response, meta_decision, task):
    if task == "gsm":
        feedback_prompt = (
                "Your answer to the following question was reviewed and marked as incorrect by the meta-reviewer.\n\n"
                f"Question: {user_query}\n\n"
                f"Your original answer: {author_response}\n\n"
                "The meta-reviewer has provided the following feedback:\n\n"
                f"{meta_decision}\n\n."
                "You must consider the meta-reviewer's suggestions seriously and revise your answer accordingly.\n\n"
                "Make sure to state your thoughts and new answer with this format:\n"
                "Thoughts: [your step-by-step computation process]\n"
                "Answer: [the final numerical answer]\n"
                "Your final answer must be a single numerical number at the end of the response.\n\n"
            )
    return feedback_prompt


def run_review_pipeline(user_query, task: str, n_reviewers=3, verbosity=0):
    author = create_author_agent()
    reviewers = create_reviewer_agents(n_reviewers)
    meta = create_meta_reviewer_agent()

    # Step 1: Author answers
    author_input = construct_author_prompt(user_query, task)
    author_response = author.run(author_input)
    if verbosity:
        print("\n=== Author's Answer ===\n", author_response)

    # Step 2: Reviewers critique
    review_responses = []
    for reviewer in reviewers:
        review_input = construct_reviewer_prompt(user_query, author_response, task)
        review = reviewer.run(review_input)
        review_responses.append(review)
        if verbosity:
            print(f"\n--- {reviewer.name} Review ---\n{review}")

    # Step 3: Meta-review
    combined_reviews = "\n\n".join(
        [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
    )
    meta_input = construct_meta_prompt(user_query, author_response, combined_reviews, task)
    meta_decision = meta.run(meta_input)
    if verbosity:
        print("\n=== Meta-Reviewer Final Decision ===\n", meta_decision)
        print("\n")

    # Additional step: build a dictionary to save the review process
    review_history = {
        "author_response": author_response,
        # "review1": review_responses[0], "review2": review_responses[1], "review3": review_responses[2],
        "meta_review": meta_decision
    }
    for i, review in enumerate(review_responses):
        review_history[f"review{i+1}"] = review

    # Step 4: Send feedback or return final answer
    decision = extract_math_decision(meta_decision)
    author_answer = extract_pred_answer(author_response)
    if decision == "wrong" or decision == "Wrong":
        feedback_input = construct_feedback_prompt(user_query, author_response, meta_decision, task)
        author_rebuttal = author.run(feedback_input)
        if verbosity:
            print("\n=== Author's new answer ===\n", author_rebuttal)
        review_history['author_rebuttal'] = author_rebuttal

    # Additional step: Compute total tokens used across all agents
    agents = [author, *reviewers, meta]
    total_tokens = sum(agent.total_tokens for agent in agents)
    review_history["total_tokens"] = total_tokens
    return review_history