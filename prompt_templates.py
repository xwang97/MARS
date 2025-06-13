class PromptBuilder:
    def __init__(self, task="gsm"):
        self.task = task

    def construct_author_prompt(self, user_query):
        if self.task == "gsm":
            author_prompt = (
                "You are a math assistant. Please help to solve the following math problem:\n"
                f"{user_query}\n\n"
                "Give your thoughts about the computation steps and the final numerical answer in the following format:\n"
                "Thoughts: [your step-by-step computation process with immediate results]\n"
                "Answer: [the final numerical answer]\n\n"
                "Your final answer must be a single numerical number at the end of the response.\n\n"
            )
        if self.task == "mmlu":
            # Add your code here
            author_prompt = (
                
            )
        return author_prompt

    def construct_reviewer_prompt(self, user_query, author_response):
        output_format = (
            "---\n\n"
            "Your output format must be:\n\n"
            "Decision: [right | wrong]  \n"
            "Confidence: [1–5] (5 = highest confidence)  \n"
            "Justification: [reasons or author mistakes supporting your decision] \n"
            "---\n\n"
        )
        if self.task == "gsm":
            reviewer_prompt = (
                "You are a reviewer. The author has submitted the following answer to a math problem:\n\n"
                f"Question: {user_query}\n\n"
                f"Answer: {author_response}\n\n"
                "Please evaluate the correctness of the author's response. Follow the instructions and format strictly:\n\n"
                f"{output_format}"
            )
        if self.task == "mmlu":
            # Add your code here
            reviewer_prompt = (
                
            )
        return reviewer_prompt

    def construct_meta_prompt(self, user_query, author_response, combined_reviews):
        output_format = (
            "Decision: [right | wrong]\n"
            "Justification: [reasons of your decision]\n"
            "Suggestions: [your suggestions for updating the answer]\n"
        )
        if self.task == "gsm":
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
        if self.task == "mmlu":
            # Add your code here
            meta_prompt = (
                
            )
        return meta_prompt

    def construct_feedback_prompt(self, user_query, author_response, meta_decision):
        if self.task == "gsm":
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
        if self.task == "mmlu":
            # Add your code here
            feedback_prompt = (
                
            )
        return feedback_prompt

    def construct_initial_prompt(self, user_query):
        return self.construct_author_prompt(user_query)

    def construct_reflection_prompt(self, user_query, response):
        if self.task == "gsm":
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
        if self.task == "mmlu":
            # Add your code here
            reflection_prompt = (
                
            )
        return reflection_prompt

    def construct_debate_prompt(self, other_agents_responses, user_query, response_idx):
        if self.task == "gsm":
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
        if self.task == "mmlu":
            # Add your code here
        return {"role": "user", "content": prompt}
