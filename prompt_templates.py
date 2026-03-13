class PromptBuilder:
    def __init__(self, task="gsm"):
        self.task = task
        self.math_tasks = ["gsm", "ciar", "gsm_hard", "svamp"]
        self.qa_tasks = ["mmlu", "gpqa", "stg", "mmlu_pro"]

    def construct_author_prompt(self, user_query):
        if self.task in self.math_tasks:
            author_prompt = {
                "role": "user",
                "content": (
                    "You are a math assistant. Please help to solve the following math problem:\n"
                    f"{user_query}\n\n"
                    "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
                    "Thoughts: [your step-by-step computation process with immediate results]\n"
                    "Answer: [the final numerical answer]\n\n"
                    "Your final answer must be a single numerical number at the end of the response.\n\n"
                )
            }
        if self.task in self.qa_tasks:
            author_prompt = {
                "role": "user",
                "content": (
                    "You are an assistant. Please help to solve the following problem:\n"
                    f"{user_query}\n\n"
                    "Give your thoughts about the question and the final answer in the following format:\n"
                    "Thoughts: [Analysis of the question and options]\n"
                    "Answer: [the final single captial letter answer in the form (X). X is chosed from [A,B,C,D]]\n\n"
                    "Your final answer must be a single capital letter in the form (X) at the end of the response.. X is from [A,B,C,D].\n\n"
                )
            }
        return author_prompt

    def construct_reviewer_prompt(self, user_query, author_response):
        # --- COMMON "SOLVE FIRST" HEADER ---
        base_instruction = (
            "You are an objective reviewer. The author has submitted an answer to a problem.\n"
            "Your goal is to check for correctness, NOT to find faults where none exist.\n"
            "INSTRUCTIONS:\n"
            "1. Ignore the Author's answer initially.\n"
            "2. Solve the problem yourself from scratch in the 'My Independent Solution' section.\n"
            "3. Only AFTER you solve it, compare your result with the Author's.\n"
        )
        
        if self.task in self.math_tasks:
            output_format = (
                "---\n"
                "OUTPUT FORMAT:\n"
                "My Independent Solution: [Solve the problem step-by-step yourself here]\n"
                "Comparison: [Compare your result with the Author's result]\n"
                "Decision: [right | wrong] (Only say 'wrong' if the Author's final value is significantly different from yours)\n"
                "Confidence: [1–5]\n"
                "Justification: [Explain strictly based on math. If the Author is right, say so.]\n"
                "Final Answer: [Your final calculated answer here]\n"
                "---\n"
            )
            criteria = "Check for arithmetic accuracy and logical consistency."

        if self.task in self.qa_tasks:
            output_format = (
                "---\n"
                "OUTPUT FORMAT:\n"
                "My Independent Analysis: [Analyze the question and evaluate each option A, B, C, D]\n"
                "My Selected Option: [e.g., (A)]\n"
                "Comparison: [Does your option match the Author's?]\n"
                "Decision: [right | wrong] (Vote 'right' if the Author's option matches yours)\n"
                "Confidence: [1–5]\n"
                "Justification: [Explain why the Author's reasoning is correct or incorrect]\n"
                "Final Answer: [Your selected option, e.g. (A)]\n"
                "---\n"
            )
            criteria = "Check for factual correctness and reasoning. Verify why the other options are wrong."
        reviewer_prompt = (
            f"{base_instruction}\n"
            f"Criteria: {criteria}\n\n"
            f"Question: {user_query}\n\n"
            f"Author's Answer: {author_response}\n\n"
            f"{output_format}"
        )
        return reviewer_prompt

    def construct_meta_prompt(self, user_query, author_response, combined_reviews):
        score_instruction = (
            "NOTE ON RELIABILITY SCORES:\n"
            "- Each Reviewer provides a 'Reliability Score' (0.0 to 1.0) based on their internal model uncertainty.\n"
            "- High Score (>0.8): The Reviewer is mathematically confident. Trust their specific calculations/facts.\n"
            "- Low Score (<0.6): The Reviewer is uncertain or confused. You should be skeptical of their critique, especially if they disagree with a High-Score reviewer.\n"
        )
        # --- MATH: Verification + Suggestions ---
        if self.task in self.math_tasks:
            specific_instructions = (
                "CRITICAL INSTRUCTIONS:\n"
                f"{score_instruction}\n"
                "1. Reviewers might be wrong. If a Reviewer claims the math is wrong but provides a vague alternative, trust your own judgment.\n"
                "2. Verify the final values yourself by substituting them back into the original question.\n"
            )
            output_format = (
                "Output Format:\n"
                "Verification: [Substitute the Author's answer into the question. Does it work?]\n"
                "Decision: [right | wrong]\n"
                "Justification: [Reasoning for your decision]\n"
                "Suggestions: [If wrong, specific guidance on which step to fix. If right, leave blank.]\n"
                "Final Answer: [The correct answer]\n"
            )

        # --- QA: Evidence + Suggestions ---
        elif self.task in self.qa_tasks:
            specific_instructions = (
                "CRITICAL INSTRUCTIONS:\n"
                f"{score_instruction}\n"
                "1. Do not just count votes. Evaluate the *reasoning* and *evidence* provided by each Reviewer.\n"
                "2. If the Author's reasoning is sound and Reviewers are nitpicking, favor the Author.\n"
            )
            output_format = (
                "Output Format:\n"
                "Evidence Evaluation: [Briefly weigh the arguments from the Author vs Reviewers]\n"
                "Decision: [right | wrong]\n"
                "Justification: [Reasoning for your decision]\n"
                "Suggestions: [If wrong, specific guidance on what facts/logic to revisit. If right, leave blank.]\n"
                "Final Answer: [The correct option in form (X)]\n"
            )
        
        meta_prompt = (
            "You are the Meta-Reviewer. You have received an answer from an Author and critiques from Reviewers.\n\n"
            f"Question: {user_query}\n\n"
            f"Author's Answer: {author_response}\n\n"
            "--- Reviewer Feedback ---\n"
            f"{combined_reviews}\n\n"
            f"{specific_instructions}\n"
            f"{output_format}\n"
        )
        return meta_prompt

    def construct_feedback_prompt(self, meta_decision, status="wrong"):
        """
        Dynamic prompt that handles both correction (if wrong) and 
        optimization/verification (if right).
        """
        if status == "right":
            # Scenario: Answer is accepted, but we force another round for stability.
            instruction = (
                "Your answer was reviewed and marked as CORRECT.\n"
                "However, we are performing one final robustness check.\n"
                "INSTRUCTIONS:\n"
                "1. Review your logic one last time to ensure no subtle bugs exist.\n"
                "2. If you are 100% certain, you may output the exact same answer.\n"
                "3. If you find a small improvement for clarity, apply it."
            )
        else:
            # Scenario: Answer was rejected.
            if self.task in self.math_tasks:
                focus = "calculation error"
            else:
                focus = "factual or reasoning error"
            
            instruction = (
                "Your answer was reviewed and marked as INCORRECT.\n"
                f"1. If you made a {focus}, fix it.\n"
                "2. HOWEVER, if you are confident your original answer was correct and the reviewers are mistaken, YOU MAY STICK TO YOUR ORIGINAL ANSWER."
            )

        feedback_prompt = (
            f"--- Reviewer Feedback ---\n{meta_decision}\n\n"
            f"{instruction}\n\n"
            "Provide your final reasoning and answer.\n"
        )
        return {"role": "user", "content": feedback_prompt}
    
    # 2. NEW: Update the Reviewer Loop Prompt (Reviewer Context)
    def construct_reviewer_update_prompt(self, previous_status, author_logic, current_answer):
        """
        Tells the reviewer WHY they are seeing this answer again.
        """
        if previous_status == "right":
            # Context: Stability Check
            context_msg = (
                "The previous answer was accepted as correct. We are now doing a robustness check.\n"
                "The Author has reviewed their work and decided to stick with (or slightly refine) their answer.\n"
                "YOUR JOB: Be extremely critical. Do not just agree because it was accepted before. Check for edge cases."
            )
        else:
            # Context: Revision
            context_msg = (
                "The previous answer was rejected. The author has revised the answer based on feedback.\n"
                "YOUR JOB: Check if the previous concerns were addressed and if new errors were introduced."
            )

        prompt = (
            f"{context_msg}\n\n"
            f"--- Author's Logic/Rebuttal ---\n{author_logic}\n\n"
            f"--- Current Answer ---\n{current_answer}\n\n"
            "Please evaluate this answer."
        )
        return {"role": "user", "content": prompt}

    def construct_initial_prompt(self, user_query):
        return self.construct_author_prompt(user_query)["content"]

    def construct_reflection_prompt(self, user_query, response):
        if self.task in ["gsm", "ciar", "gsm_hard", "svamp"]:
            reflection_prompt = (
                "You wrote the following response to a math problem:\n\n"
                f"Qustion: {user_query}\n\n"
                f"Answer: {response}\n\n"
                "Carefully review your own answer. Are there any mistakes, inconsistencies, or calculation errors?\n"
                "If yes, explain the problems and revise your answer accordingly. If not, confirm and repeat your initial answer."
                "Your final response must follow this format:\n"
                "Mistakes (if any): \n\n"
                "Answer: [the final single numerical answer]\n\n"
            )
        if self.task in ["mmlu", "gpqa", "stg", "mmlu_pro"]:
            # Add your code here
            reflection_prompt = (
                "You wrote the following response to a problem:\n\n"
                f"Qustion: {user_query}\n\n"
                f"Answer: {response}\n\n"
                "Carefully review your own answer. Are there any mistakes or thoughts not grounded in the given problem or known facts?\n"
                "If yes, explain the problems and revise your answer accordingly. If not, confirm and repeat your initial answer."
                "Your final response must follow this format:\n"
                "Mistakes (if any): \n\n"
                "Answer: [the final single captial letter answer in the form (X). X is chosed from [A,B,C,D]]\n\n"
            )
        return reflection_prompt

    def construct_debate_prompt(self, other_agents_responses, user_query, response_idx):
        if self.task in ["gsm", "ciar", "gsm_hard", "svamp"]:
            if not other_agents_responses:
                return {
                    "role": "user",
                    "content": (
                        "You are a math assistant. Please help to solve the following math problem:\n"
                        f"{user_query}\n\n"
                        "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
                        "Thoughts: [your step-by-step computation process with immediate results]\n"
                        "Answer: [the final numerical answer]\n\n"
                        "Your final answer must be a single numerical number at the end of the response.\n\n"
                    )
                }

            prompt = "These are the solutions to the problem from other agents:\n"
            for history in other_agents_responses:
                response = history[response_idx]["content"]
                prompt += f"\n\nOne agent solution: ```{response}```"

            prompt += (
                "\n\nUsing the solutions from other agents as additional information, can you provide your final answer to the math problem?\n"
                "Make sure to state your thoughts and new answer with this format:\n"
                "Thoughts: [your step-by-step computation process]\n"
                "Answer: [the final numerical answer]\n"
                "Your final answer must be a single numerical number at the end of the response.\n\n"
            )
        
        if self.task in ["mmlu", "gpqa", "stg", "mmlu_pro"]:
            if not other_agents_responses:
                return {
                    "role": "user",
                    "content": (
                        "You are an assistant. Please help to solve the following problem:\n"
                        f"{user_query}\n\n"
                        "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
                        "Thoughts: [your step-by-step thinking process with immediate results]\n"
                        "Answer: [the final single captial letter answer in the form (X). X is chosed from [A,B,C,D]]\n\n"
                        "Your final answer must be a single captial letter at the end of the response.\n\n"
                    )
                }
        
            prompt = "These are the solutions to the problem from other agents:\n"
            for history in other_agents_responses:
                response = history[response_idx]["content"]
                prompt += f"\n\nOne agent solution: ```{response}```"
        
            prompt += (
                "\n\nUsing the solutions from other agents as additional information, can you provide your final answer to the problem?\n"
                "Make sure to state your thoughts and new answer with this format:\n"
                "Thoughts: [your step-by-step thinking process]\n"
                "Answer: [the final capital letter answer]\n"
                "Your final answer must be a single capital letter at the end of the response.\n\n"
            )
        return {"role": "user", "content": prompt}
