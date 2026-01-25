from openai import OpenAI
from utils import get_api_key
import yaml
import random
import numpy as np


# ======== API key Setup ==============
openai_api_key = get_api_key("../openai_api_key.txt")
nvidia_api_key = get_api_key("../nvidia_api_key.txt")

# ======== LLM Setup ==============
# Load the YAML config file
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
author_llm = config['author_llm']
reviewer_llms = config['reviewer_llms']
meta_llm = config['meta_llm']


# ======= Definition of the OpenAI agent class ================
class OpenAIAgent:
    def __init__(self, name, model="gpt-3.5-turbo", temperature=1.0, seed=None):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.seed = seed  # If None, we will generate one per call
        self.openai_sdk = True
        self.client = None
        if "gpt" in model:
            self.client = OpenAI(api_key=openai_api_key)
        elif "llama" in model:
            # self.client = Cerebras(api_key=cerebras_api_key)
            self.client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        elif "qwen" in model or "gemma" in model or "mistral" in model:
            self.client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_openai(messages)["content"]  # Return string for backward compatibility
        elif isinstance(prompt, list):
            return self._call_openai(prompt)  # Return dict for debate/chat-style use

    def _call_openai(self, messages: list[dict]) -> dict:
        current_call_seed = self.seed if self.seed is not None else random.randint(0, 1000000)
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "seed": current_call_seed,  # <--- Pass the explicit seed
            "logprobs": True,
        }
        if not self.openai_sdk:
            response = self.client.chat.complete(**kwargs)
        else:
            response = self.client.chat.completions.create(**kwargs)
        # --- METHOD 1: AVG LOGPROB ---
        confidence_score = 0.5 # Default
        if response.choices[0].logprobs:
            # Extract list of logprobs
            token_logprobs = [
                t.logprob for t in response.choices[0].logprobs.content 
                if t.logprob is not None
            ]
            if token_logprobs:
                avg_logprob = np.mean(token_logprobs)
                confidence_score = np.exp(avg_logprob) # Normalize to 0-1
        # Track token usage
        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens
            self.token_log.append({
                "agent": self.name,
                "model": self.model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "messages_preview": messages[-1]["content"][:100],  # optional
                "used_seed": current_call_seed,
            })
        return {
            "role": "assistant",
            "content": response.choices[0].message.content.strip(),
            "confidence_score": confidence_score,
        }


# ======== Define the review related agents ==================
def create_author_agent(name="Author", model=None):
    return OpenAIAgent(
        name=name,
        model=author_llm if model is None else model
    )


def create_reviewer_agents(num_reviewers: int = 3, model=None):
    reviewers = []
    for i in range(num_reviewers):
        # llm = random.choice(reviewer_llms)
        llm = reviewer_llms[i]
        reviewer = OpenAIAgent(
            name=f"Reviewer_{i+1}",
            model=llm if model is None else model
        )
        reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent(model=None):
    return OpenAIAgent(
        name="MetaReviewer",
        model=meta_llm if model is None else model
    )