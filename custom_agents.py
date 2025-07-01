from openai import OpenAI
import random
from utils import get_api_key


# ======== API key Setup ==============
openai_api_key = get_api_key("../openai_api_key_4_XiaoWang.txt")
nvidia_api_key = get_api_key("../nvidia_api_key.txt")

# ======== LLM Setup ==============
author_llm = "meta/llama-3.3-70b-instruct"
reviewer_llms = [
    "meta/llama-3.3-70b-instruct",
]
meta_llm = "meta/llama-3.3-70b-instruct"


# ======= Definition of the OpenAI agent class ================
class OpenAIAgent:
    def __init__(self, name, model="gpt-3.5-turbo"):
        self.name = name
        self.model = model
        if model[:3] == "gpt":
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key=nvidia_api_key,
            )
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_openai(messages)["content"]  # Return string for backward compatibility
        elif isinstance(prompt, list):
            return self._call_openai(prompt)  # Return dict for debate/chat-style use

    def _call_openai(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
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
                "messages_preview": messages[-1]["content"][:100]  # optional
            })
        return {
            "role": "assistant",
            "content": response.choices[0].message.content.strip()
        }


# ======== Define the review related agents ==================
def create_author_agent(name="Author"):
    return OpenAIAgent(
        name=name,
        model=author_llm,
    )


def create_reviewer_agents(num_reviewers: int = 3):
    reviewers = []
    for i in range(num_reviewers):
        llm = random.choice(reviewer_llms)
        reviewer = OpenAIAgent(
            name=f"Reviewer_{i+1}",
            model=llm,
        )
        reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent():
    return OpenAIAgent(
        name="MetaReviewer",
        model=meta_llm,
    )