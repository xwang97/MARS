from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, DuckDuckGoSearchTool
from typing import List
import random


# Adjust the logging level for smolagents
verbosity = 1

# LLM setup
author_llm = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
reviewer_llms = [
    HfApiModel(model_id="meta-llama/Llama-3.1-8B-Instruct"),
    HfApiModel(model_id="mistralai/Mistral-7B-Instruct-v0.3"),
    HfApiModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"),
    HfApiModel(),
]
meta_llm = HfApiModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096)

reviewer_styles = [
    "Be conservative. Flag anything you're unsure about. Avoid giving the benefit of the doubt.",
    "Be balanced. Use both internal consistency and known facts before making a decision.",
    "Be skeptical. Assume statements may be incorrect unless strongly supported.",
    "Be generous. Only flag as hallucinated if there is clear inconsistency or falsehood.",
    "Focus on logical coherence and internal support only. Avoid relying on world knowledge unless essential.",
    "Be thorough and critical. Use both passage and external evidence to verify claims carefully."
]


# ========== Agent Definitions ==========

def create_author_agent(agent_type="code"):
    if agent_type == "code":
        return CodeAgent(
            name="Author",
            model=author_llm,
            tools=[],  # Future: Insert tool functions here
            verbosity_level=verbosity,
            max_steps=5,
        )
    else:
        return ToolCallingAgent(
            name="Author",
            model=author_llm,
            tools=[],  # Future: Insert tool functions here
            verbosity_level=verbosity,
            max_steps=5,
        )


def create_reviewer_agents(num_reviewers: int = 3, agent_type="code"):
    reviewers = []
    for i in range(num_reviewers):
        style = random.choice(reviewer_styles)
        llm = random.choice(reviewer_llms)
        if agent_type == "code":
            reviewer = CodeAgent(
                name=f"Reviewer_{i+1}",
                model=llm,
                tools=[DuckDuckGoSearchTool(max_results=3)],
                verbosity_level=verbosity,
                max_steps=5,
            )
        else:
            reviewer = ToolCallingAgent(
                name=f"Reviewer_{i+1}",
                model=llm,
                tools=[DuckDuckGoSearchTool(max_results=3)],
                verbosity_level=verbosity,
                max_steps=5,
            )
        reviewer.style_prompt = style  # Attach for reference/logging
        reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent(agent_type="code") -> CodeAgent:
    if agent_type == "code":
        return CodeAgent(
            name="MetaReviewer",
            model=meta_llm,
            tools=[],
            verbosity_level=verbosity,
            max_steps=5,
        )
    else:
        return ToolCallingAgent(
            name="MetaReviewer",
            model=meta_llm,
            tools=[],
            verbosity_level=verbosity,
            max_steps=5,
        )
