from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool
from typing import List
import random


# Adjust the logging level for smolagents
verbosity = 1

# LLM setup
author_llm = HfApiModel()
reviewer_llms = [
    HfApiModel(model="HuggingFaceH4/zephyr-7b-alpha"),
    HfApiModel(model="mistralai/Mistral-7B-Instruct-v0.2"),
    HfApiModel(model="google/gemma-7b-it"),  # Optional: test for variety
    HfApiModel(),
]
meta_llm = HfApiModel(model="deepseek-ai/DeepSeek-R1")

reviewer_styles = [
    "Be conservative. Flag anything you're unsure about. Avoid giving the benefit of the doubt.",
    "Be balanced. Use both internal consistency and known facts before making a decision.",
    "Be skeptical. Assume statements may be incorrect unless strongly supported.",
    "Be generous. Only flag as hallucinated if there is clear inconsistency or falsehood.",
    "Focus on logical coherence and internal support only. Avoid relying on world knowledge unless essential.",
    "Be thorough and critical. Use both passage and external evidence to verify claims carefully."
]


# ========== Agent Definitions ==========

def create_author_agent() -> CodeAgent:
    return CodeAgent(
        name="Author",
        model=author_llm,
        tools=[],  # Future: Insert tool functions here
        verbosity_level=verbosity,
    )


def create_reviewer_agents(num_reviewers: int = 3) -> List[CodeAgent]:
    reviewers = []
    for i in range(num_reviewers):
        style = random.choice(reviewer_styles)
        llm = random.choice(reviewer_llms)
        print(llm)
        reviewer = CodeAgent(
            name=f"Reviewer_{i+1}",
            model=llm,
            tools=[DuckDuckGoSearchTool(max_results=3)],
            verbosity_level=verbosity,
        )
        reviewer.style_prompt = style  # Attach for reference/logging
        reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent() -> CodeAgent:
    return CodeAgent(
        name="MetaReviewer",
        model=meta_llm,
        tools=[],
        verbosity_level=verbosity,
    )
