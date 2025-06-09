import json
import random
from openai import OpenAI
from custom_agents import OpenAIAgent  # adjust to your actual file/module path

def construct_debate_prompt(other_agents_responses, question, response_idx):
    if not other_agents_responses:
        return {
            "role": "user",
            "content": (
                f"Can you solve the following math problem?\n\n{question}\n\n"
                "Explain your reasoning. Your final answer should be a single numerical number, "
                "in the form \\boxed{{answer}}, at the end of your response."
            )
        }

    prompt = "These are the solutions to the problem from other agents:\n"
    for history in other_agents_responses:
        response = history[response_idx]["content"]
        prompt += f"\n\nOne agent solution: ```{response}```"

    prompt += (
        f"\n\nUsing the solutions from other agents as additional information, "
        f"can you provide your own answer to the math problem?\n"
        f"The original problem is:\n\n{question}\n\n"
        "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end."
    )
    return {"role": "user", "content": prompt}


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_debate_pipeline(path_to_questions, api_key, num_agents=3, num_rounds=2, max_questions=100, output_path=None):
    questions = read_jsonl(path_to_questions)
    random.shuffle(questions)

    # Initialize agents
    agents = [OpenAIAgent(name=f"Agent_{i+1}", model="gpt-3.5-turbo", api_key=api_key) for i in range(num_agents)]
    results = {}

    for example in questions[:max_questions]:
        question = example["question"]
        answer = example["answer"]

        # Each agent starts with a history (chat log)
        agent_histories = [[] for _ in range(num_agents)]

        # Round 0: initial answers without seeing others
        for i in range(num_agents):
            prompt = construct_debate_prompt([], question, 0)
            agent_histories[i].append(prompt)
            response = agents[i].run(prompt)
            agent_histories[i].append(response)

        # Subsequent rounds: agents revise based on others' responses
        for r in range(1, num_rounds):
            for i in range(num_agents):
                others = agent_histories[:i] + agent_histories[i+1:]
                prompt = construct_debate_prompt(others, question, 2*r - 1)
                agent_histories[i].append(prompt)
                response = agents[i].run(agent_histories[i])
                agent_histories[i].append(response)

        # Save all agents' histories and the ground-truth answer
        results[question] = {
            "answer": answer,
            "agent_histories": agent_histories
        }

    # Save to disk if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results, agents