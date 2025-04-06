from agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent


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