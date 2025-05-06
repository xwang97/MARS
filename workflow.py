from agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent


def run_detection_pipeline(sentence: str, passage: str):
    # reviewers = create_reviewer_agents(agent_type="toolcall")
    # meta = create_meta_reviewer_agent(agent_type="toolcall")
    reviewers = create_reviewer_agents()
    meta = create_meta_reviewer_agent()

    print("\n[📥 Sentence for Evaluation]:")
    print(sentence)
    print("\n[📘 Supporting Passage Context]:")
    print(passage)

    # Step 1: Reviewers analyze the sentence with passage context
    review_responses = []
    for reviewer in reviewers:
        review_input = (
            "You are a hallucination reviewer.\n"
            "Your task is to determine whether the following sentence is factual. You can only answer factual if it meets all of these criteria: \n"
            "1. Consistent with known facts (general world knowledge).\n"
            "2. Consistent with the passage.\n\n"
            "Always keep these things in mind: \n"
            "1. Remember to start by using web_search tool to search on the internet.\n"
            "2. You can only use the passage for interpreting ambiguous terms or pronouns. The passage itself may not be factually correct. " 
            "You must not conclude the sentence is true because it is consistent with the passage.\n\n"
            "Your output must follow this structure exactly:\n\n"
            "Decision: [factual | non-factual]\n"
            "Confidence: [1-5]\n"
            "Reasons:\n"
            "- Reason 1\n"
            "- Reason 2 (optional)\n"
            "- Reason 3 (optional)\n\n"
            "📘 Passage:\n"
            f"{passage}\n\n"
            "📥 Sentence:\n"
            f"{sentence}\n\n"
            "Remeber the decision must be non-factual if any one of the criteria is violated. Now make your classification and explain your reasoning following the structure above."
        )

        review = reviewer.run(review_input)
        review_responses.append(review)
        print(f"\n--- {reviewer.name} Review ---\n{review}")

    # Step 2: Meta-reviewer synthesizes and decides
    combined_reviews = "\n\n".join(
        [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
    )
    meta_input = (
        "You are a meta-reviewer. Your task is to determine whether a sentence is hallucinated, based on two sources:\n"
        "1. The reviews from multiple agents (which include their decision, confidence, and reasoning)\n"
        "2. Your own general world knowledge and critical judgment\n\n"
        "Some reviewers may use web search to support their reasoning. "
        "You should trust well-sourced search-backed justifications over unsupported guesses or vague reasoning. "
        "If reviewers cite external evidence, weigh that highly in your final judgment.\n"
        "The sentence originates from the passage and may be interpreted using it (e.g., for resolving references). "
        "However, the passage itself is not guaranteed to be true. You must also consider whether the sentence presents "
        "fabricated or incorrect information based on general known facts.\n\n"
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
        "📘 Passage:\n"
        f"{passage}\n\n"
        "📥 Sentence:\n"
        f"{sentence}\n\n"
        "--- Reviewer Feedback ---\n"
        f"{combined_reviews}\n\n"
        "📢 Provide your final classification using the structure above. Be concise, clear, and well-justified."
    )

    final_decision = meta.run(meta_input)
    print("\n=== 🧠 Meta-Reviewer Final Judgment ===\n", final_decision)
    return final_decision


def run_review_pipeline(user_query: str):
    author = create_author_agent()
    reviewers = create_reviewer_agents()
    meta = create_meta_reviewer_agent()

    # Step 1: Author answers
    author_input = (
        f"You are an assistant. Please answer the following question clearly and factually:\n\n{user_query}"
    )
    author_response = author.run(author_input)
    print("\n=== Author's Answer ===\n", author_response)

    # Step 2: Reviewers critique
    review_responses = []
    for reviewer in reviewers:
        review_input = (
            f"You are a reviewer. The author answered this question:\n\n"
            f"Question: {user_query}\n\n"
            f"Answer: {author_response}\n\n"
            f"Review the answer for hallucinations or inaccuracies."
        )
        review = reviewer.run(review_input)
        review_responses.append(review)
        print(f"\n--- {reviewer.name} Review ---\n{review}")

    # Step 3: Meta-review
    meta_input = (
        f"You are the meta-reviewer. The author answered:\n\n{author_response}\n\n"
        f"Reviews:\n" + "\n\n".join(
            [f"{r.name}:\n{r.run(f'Review of {author_response}')}" for r in reviewers]
        ) + "\n\nMake a final judgment: is the author's answer trustworthy?"
    )
    final_decision = meta.run(meta_input)
    print("\n=== Meta-Reviewer Final Decision ===\n", final_decision)