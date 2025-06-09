from openai import OpenAI
import random
from utils import get_openai_api_key


# ======= Definition of the OpenAI agent class ================
class OpenAIAgent:
    def __init__(self, name, model="gpt-3.5-turbo", api_key=None):
        self.name = name
        self.model = model
        self.client = OpenAI(api_key=api_key)

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


# ======== LLM Setup ==============
openai_api_key = get_openai_api_key("../openai_api_key_4_XiaoWang.txt")
author_llm = "gpt-4o-mini"
reviewer_llms = ["gpt-4o-mini"]
meta_llm = "gpt-4o-mini"


# ======== Define the review related agents ==================
def create_author_agent():
    return OpenAIAgent(
        name="Author",
        model=author_llm,
        api_key=openai_api_key
    )


def create_reviewer_agents(num_reviewers: int = 3):
    reviewers = []
    for i in range(num_reviewers):
        llm = random.choice(reviewer_llms)
        reviewer = OpenAIAgent(
            name=f"Reviewer_{i+1}",
            model=llm,
            api_key=openai_api_key
        )
        reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent():
    return OpenAIAgent(
        name="MetaReviewer",
        model=meta_llm,
        api_key=openai_api_key
    )
