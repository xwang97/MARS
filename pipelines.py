from custom_agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent
from utils import extract_math_decision, extract_pred_answer
from prompt_templates import PromptBuilder


class PipelineRunner:
    def __init__(self, task="gsm", agent_backend="openai"):
        self.task = task
        self.templates = PromptBuilder(task=task)

    def run_marvel_pipeline(self, user_query, n_reviewers=3, verbosity=0):
        author = create_author_agent()
        reviewers = create_reviewer_agents(n_reviewers)
        meta = create_meta_reviewer_agent()
        author_history = []

        # Step 1: Author answers
        author_input = self.templates.construct_author_prompt(user_query)
        author_history.append(author_input)
        author_response = author.run(author_history)
        author_history.append(author_response)
        if verbosity:
            print("\n=== Author's Answer ===\n", author_response["content"])

        # Step 2: Reviewers critique
        review_responses = []
        for reviewer in reviewers:
            review_input = self.templates.construct_reviewer_prompt(user_query, author_response["content"])
            review = reviewer.run(review_input)
            review_responses.append(review)
            if verbosity:
                print(f"\n--- {reviewer.name} Review ---\n{review}")

        # Step 3: Meta-review
        combined_reviews = "\n\n".join(
            [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
        )
        meta_input = self.templates.construct_meta_prompt(user_query, author_response["content"], combined_reviews)
        meta_decision = meta.run(meta_input)
        if verbosity:
            print("\n=== Meta-Reviewer Final Decision ===\n", meta_decision)
            print("\n")

        # Additional step: build a dictionary to save the review process
        review_history = {"author_response": author_response["content"]}
        for i, review in enumerate(review_responses):
            review_history[f"review{i+1}"] = review
        review_history["meta_review"] = meta_decision

        # Step 4: Send feedback or return final answer
        decision = extract_math_decision(meta_decision)
        if decision.lower() == "wrong":
            feedback_input = self.templates.construct_feedback_prompt(meta_decision)
            author_history.append(feedback_input)
            author_rebuttal = author.run(author_history)
            if verbosity:
                print("\n=== Author's new answer ===\n", author_rebuttal["content"])
            review_history['author_rebuttal'] = author_rebuttal["content"]

        # Additional step: Compute total tokens used across all agents
        agents = [author, *reviewers, meta]
        total_tokens = sum(agent.total_tokens for agent in agents)
        review_history["total_tokens"] = total_tokens
        return review_history

    def run_self_reflection_pipeline(self, user_query, verbosity=0):
        agent = create_author_agent()
        # Step 1: Initial answer
        author_input = self.templates.construct_initial_prompt(user_query)
        response = agent.run(author_input)
        if verbosity:
            print("\n=== Initial Answer ===\n", response)

        # Step 2: Self-reflection
        reflection_prompt = self.templates.construct_reflection_prompt(user_query, response)
        reflection = agent.run(reflection_prompt)
        if verbosity:
            print("\n=== Final answer after self-reflection ===\n", reflection)
        reflection_history = {"response": response, "reflection": reflection, "total_tokens": agent.total_tokens}
        return reflection_history

    def run_debate_pipeline(self, user_query, num_agents=3, num_rounds=2, verbosity=0) -> list[list[dict]]:
        agents = [
            create_author_agent(name=f"Agent_{i+1}")
            for i in range(num_agents)
        ]
        agent_histories = [[] for _ in range(num_agents)]

        # Round 0: each agent answers independently
        for i in range(num_agents):
            prompt = self.templates.construct_debate_prompt([], user_query, response_idx=0)
            agent_histories[i].append(prompt)
            response = agents[i].run(agent_histories[i])
            agent_histories[i].append(response)
            if verbosity:
                print(f"\n=== Round 0 Agent {i+1} Answer ===\n", response["content"])

        # Rounds >= 1: agents revise based on others
        for r in range(1, num_rounds):
            for i in range(num_agents):
                other_histories = agent_histories[:i] + agent_histories[i+1:]
                prompt = self.templates.construct_debate_prompt(other_histories, user_query, response_idx=2*r - 1)
                agent_histories[i].append(prompt)
                response = agents[i].run(agent_histories[i])
                agent_histories[i].append(response)
                if verbosity:
                    print(f"\n=== Round {r} Agent {i+1} Answer ===\n", response["content"])
        total_tokens = sum(agent.total_tokens for agent in agents)

        return agent_histories, total_tokens  # List of message histories per agent