from custom_agents import create_author_agent, create_reviewer_agents, create_meta_reviewer_agent
from utils import extract_meta_decision, extract_pred_answer
from prompt_templates import PromptBuilder


class PipelineRunner:
    def __init__(self, task="gsm", model=None):
        self.task = task
        self.templates = PromptBuilder(task=task)
        self.model = model

    def run_mars_pipeline(self, user_query, n_reviewers=2, max_rounds=2, verbosity=0):
        # 1. Initialize Agents
        author = create_author_agent(model=self.model)
        reviewers = create_reviewer_agents(n_reviewers, model=self.model)
        meta = create_meta_reviewer_agent(model=self.model)

        # 2. Setup History Tracking
        author_history = []
        # We need a separate conversation history for EACH reviewer
        reviewer_histories = [[] for _ in range(n_reviewers)] 
        full_history = {
            "user_query": user_query,
            "rounds": [],
            "final_status": "unresolved"
        }

        # 3. Step 1: Author's First Draft
        author_input = self.templates.construct_author_prompt(user_query)
        author_history.append(author_input)
        
        author_response = author.run(author_history)
        author_history.append(author_response)
        current_answer_content = author_response["content"]
        
        # Variable to store the author's rebuttal logic to show reviewers in the next round
        latest_rebuttal_logic = "This is the initial answer." 
        last_round_decision = "wrong"
        if verbosity:
            print(f"\n=== Round 0: Author's Initial Answer ===\n{current_answer_content}")

        # 4. The Review Loop
        for round_idx in range(1, max_rounds + 1):
            if verbosity:
                print(f"\n\n--- Starting Review Round {round_idx} ---")

            round_data = {
                "round": round_idx,
                "input_answer": current_answer_content,
                "reviews": [],
                "meta_review": None
            }

            # --- A. Reviewers Critique (Stateful) ---
            review_responses = []
            for i, reviewer in enumerate(reviewers):               
                # Logic: Construct the prompt based on whether it is the first round or a follow-up
                if round_idx == 1:
                    # First time: Standard Reviewer Prompt
                    prompt_content = self.templates.construct_reviewer_prompt(user_query, current_answer_content)
                    # Initialize this reviewer's history
                    reviewer_histories[i] = [{"role": "user", "content": prompt_content}]
                else:
                    # Round 2+: Dynamic Update Message
                    # We need the 'decision' from the previous round to set the status
                    # If this is round 2, we look at what happened at the end of round 1.
                    # (You need to track 'last_round_decision' variable in your loop)                    
                    prev_status = last_round_decision # Use the tracked decision from the previous round
                    
                    # Use the NEW method
                    update_msg_dict = self.templates.construct_reviewer_update_prompt(
                        previous_status=prev_status, # "right" or "wrong"
                        author_logic=latest_rebuttal_logic, 
                        current_answer=current_answer_content
                    )
                    reviewer_histories[i].append(update_msg_dict)

                # RUN the reviewer (passing the list triggers the chat-history mode in your Agent class)
                review_response_dict = reviewer.run(reviewer_histories[i])
                review_content = review_response_dict["content"]
                # Extract confidence score if available
                confidence_score = review_response_dict.get("confidence_score", 0.5)
                formatted_review_str = (
                    f"Reliability Score: {confidence_score:.2f} / 1.0\n"
                    f"{review_content}"
                )
                review_responses.append(formatted_review_str)

                # Append the *Reviewer's own output* to their history so they remember what they said
                reviewer_histories[i].append(review_response_dict)
                
                # Save string content for the Meta-Reviewer
                round_data["reviews"].append({f"reviewer_{i+1}": formatted_review_str})                
                if verbosity:
                    print(f"\n[Round {round_idx}] {reviewer.name}:\n{review_content}")

            # --- B. Meta-Reviewer Decision (Remains mostly stateless per round) ---
            combined_reviews = "\n\n".join(
                [f"{reviewers[i].name}:\n{review_responses[i]}" for i in range(len(reviewers))]
            )
            meta_input = self.templates.construct_meta_prompt(user_query, current_answer_content, combined_reviews)
            meta_decision = meta.run(meta_input)            
            round_data["meta_review"] = meta_decision
            if verbosity:
                print(f"\n[Round {round_idx}] Meta-Reviewer Decision:\n{meta_decision}")

            # --- C. Check Decision ---
            decision = extract_meta_decision(meta_decision).lower() # Normalize to 'right'/'wrong'
            last_round_decision = decision # Update for next round's reviewer prompts
            # Update status for the record
            round_data["decision"] = decision
            if decision == "right":
                full_history["final_status"] = "accepted"
                # DO NOT BREAK here. We continue to max_rounds.
            if verbosity:
                 print(f"\n[Round {round_idx}] Decision: {decision.upper()}")

            # --- D. Feedback & Revision ---
            # We only generate feedback if we haven't reached the limit
            if round_idx < max_rounds:
                
                # NEW: Pass the decision status to the prompt builder
                # If decision was "right", the prompt will be "Double check it".
                # If decision was "wrong", the prompt will be "Fix it".
                feedback_input = self.templates.construct_feedback_prompt(meta_decision, status=decision)
                
                author_history.append(feedback_input)
                author_rebuttal = author.run(author_history)
                author_history.append(author_rebuttal)
                
                # Update current content for the next loop
                current_answer_content = author_rebuttal["content"]
                latest_rebuttal_logic = author_rebuttal["content"]
                
                if verbosity:
                    print(f"\n=== Author's Revised Answer (Round {round_idx}) ===\n{current_answer_content}")
            else:
                # If we are at the last round, just mark the final status based on the LAST decision
                if decision == "right":
                    full_history["final_status"] = "accepted"
                else:
                    full_history["final_status"] = "rejected"

            full_history["rounds"].append(round_data)

        # Final Token Calculation
        agents = [author, *reviewers, meta]
        total_tokens = sum(agent.total_tokens for agent in agents)
        full_history["total_tokens"] = total_tokens
        full_history["final_answer"] = current_answer_content
        
        return full_history

    def run_single_agent_pipeline(self, user_query, verbosity=0):
        agent = create_author_agent(model=self.model)
        author_input = self.templates.construct_initial_prompt(user_query)
        response = agent.run(author_input)
        if verbosity:
            print("\n=== Agent Answer ===\n", response)
        agent_history = {"response": response, "total_tokens": agent.total_tokens}
        return agent_history

    def run_self_reflection_pipeline(self, user_query, verbosity=0):
        agent = create_author_agent(model=self.model)
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

    def run_self_consistency_pipeline(self, user_query, num_samples=3, verbosity=0):
        agent = create_author_agent(model=self.model)
        responses = []
        for i in range(num_samples):
            author_input = self.templates.construct_initial_prompt(user_query)
            response = agent.run(author_input)
            responses.append(response)
            if verbosity:
                print(f"\n=== Sample {i+1} Answer ===\n", response)
        sc_history = {"responses": responses, "total_tokens": agent.total_tokens}
        return sc_history

    def run_debate_pipeline(self, user_query, num_agents=3, num_rounds=2, verbosity=0) -> list[list[dict]]:
        agents = [
            create_author_agent(name=f"Agent_{i+1}", model=self.model)
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