class PromptBuilder:
    def __init__(self, task="gsm"):
        self.task = task

    def construct_author_prompt(self, user_query):
        if self.task in ["gsm", "ciar", "gsm_hard"]:
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
        if self.task == "mmlu" or self.task=="gpqa":
            author_prompt = {
                "role": "user",
                "content": (
                    "You are an assistant. Please help to solve the following problem:\n"
                    f"{user_query}\n\n"
                    "Give your thoughts about the question and the final answer in the following format:\n"
                    "Thoughts: [your thoughts with immediate results]\n"
                    "Answer: [the final single captial letter answer in the form (X). X is chosed from [A,B,C,D]]\n\n"
                    "Your final answer must be a single capital letter in the form (X) at the end of the response.. X is from [A,B,C,D].\n\n"
                )
            }
        return author_prompt

    def construct_reviewer_prompt(self, user_query, author_response):
        output_format = (
            "---\n\n"
            "Your output format must be:\n\n"
            "Decision: [right | wrong]  \n"
            "Confidence: [1–5] (5 = highest confidence)  \n"
            "Justification: [reasons or author mistakes supporting your decision] \n"
            "Answer: [your recommended answer]"
            "---\n\n"
        )
        if self.task in ["gsm", "ciar", "gsm_hard"]:
            reviewer_prompt = (
                "You are a reviewer. The author has submitted the following answer to a math problem:\n\n"
                f"Question: {user_query}\n\n"
                f"Answer: {author_response}\n\n"
                "Please evaluate the correctness of the author's response. Follow the instructions and format strictly:\n\n"
                "---\n\n"
                "Evaluation criteria:\n\n"
                "1. **Consistency**: Check whether each step in the author's thoughts is consistent with the original problem.\n"
                "2. **Accuracy**: Check whether each computation gets the correct result.\n"
                f"{output_format}"
            )
        if self.task == "mmlu" or self.task=="gpqa":
            reviewer_prompt = (
                "You are a reviewer. The author has submitted the following answer to a problem:\n\n"
                f"Question: {user_query}\n\n"
                f"Answer: {author_response}\n\n"
                "Please evaluate the correctness of the author's response. Follow the instructions and format strictly:\n\n"
                "---\n\n"
                "Evaluation criteria:\n\n"
                "**Faithfulness**: check whether the author's answers is consistent with facts you have known.\n"
                f"{output_format}"
            )
        return reviewer_prompt

    def construct_meta_prompt(self, user_query, author_response, combined_reviews):
        output_format = (
            "Decision: [right | wrong]\n"
            "Justification: [reasons of your decision]\n"
            "Suggestions: [your suggestions for updating the answer, only needed when decision is wrong]\n"
        )
        if self.task in ["gsm", "ciar", "gsm_hard"]:
            meta_prompt = (
                "You are the meta-reviewer. The author has submitted an answer to a math problem.\n\n"
                f"Question: {user_query}\n\n"
                f"Answer: {author_response}\n\n"
                "You must decide whether the answer is correct based on:\n"
                "1. Your own mathematical knowledge\n"
                "2. The reviewers' comments provided below\n\n"
                "--- Reviewer Feedback ---\n"
                f"{combined_reviews}\n\n"
                "Do not only rely on the reviewers, you must also think by yourself.\n\n"
                "Provide your conclusion in the following format:\n"
                f"{output_format}"
                "Answer: [your recommended single numerical answer]\n\n"
            )
            # "Answer: [your recommended single numerical answer]\n\n"
        if self.task == "mmlu" or self.task=="gpqa":
            # Add your code here
            meta_prompt = (
                "You are the meta-reviewer. The author has submitted an answer to a problem.\n\n"
                f"Question: {user_query}\n\n"
                f"Answer: {author_response}\n\n"
                "You must decide whether the answer is correct based on both your own knowledge and the reviewers' comments below:\n\n"
                "--- Reviewer Feedback ---\n"
                f"{combined_reviews}\n\n"
                "Do not only rely on the reviewers, you must also think by yourself.\n\n"
                "Provide your conclusion in the following format:\n"
                f"{output_format}"
                "Answer: [your recommended single numerical answer]\n\n"
            )
        return meta_prompt

    def construct_feedback_prompt(self, meta_decision):
        if self.task in ["gsm", "ciar", "gsm_hard"]:
            feedback_prompt = (
                    "Your answer was reviewed and marked as incorrect by the meta-reviewer.\n\n"
                    "--- Meta-reviewer Feedback ---\n"
                    f"{meta_decision}\n\n."
                    "If you agree with the meta-reviewer's suggestions, revise your answer accordingly.\n"
                    "If you disagree, insist on your initial answer and repeat it.\n\n"
                    "Make sure to state your thoughts and final answer with this format:\n"
                    "Reasons: [your reasons of accepting or rejecting the suggestions]\n"
                    "Answer: [the final numerical answer]\n\n"
                )
        if self.task == "mmlu" or self.task=="gpqa":
            # Add your code here
            feedback_prompt = (
                    "Your answer was reviewed and marked as incorrect by the meta-reviewer.\n\n"
                    "--- Meta-reviewer Feedback ---\n"
                    f"{meta_decision}\n\n."
                    "If you strongly agree with the meta-reviewer's suggestions, revise your answer accordingly.\n"
                    "If you disagree, insist on your initial answer and repeat it.\n\n"
                    "Make sure to state your reasons and final answer with this format:\n"
                    "Reasons: [your reasons of accepting or rejecting the suggestions]\n"
                    "Answer: [the final single captial letter answer in the form (X). X is chosed from [A,B,C,D]]\n\n"
            )
        return {"role": "user", "content": feedback_prompt}

    def construct_initial_prompt(self, user_query):
        return self.construct_author_prompt(user_query)["content"]

    def construct_reflection_prompt(self, user_query, response):
        if self.task in ["gsm", "ciar", "gsm_hard"]:
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
        if self.task == "mmlu" or self.task=="gpqa":
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
        if self.task in ["gsm", "ciar", "gsm_hard"]:
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
        
        if self.task == "mmlu" or self.task=="gpqa":
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
