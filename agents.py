from smolagents import CodeAgent, HfApiModel
from typing import List

# LLM setup
llm = HfApiModel(model="HuggingFaceH4/zephyr-7b-alpha")


# ========== Agent Definitions ==========

def create_author_agent() -> CodeAgent:
    return CodeAgent(
        name="Author",
        model=llm,
        tools=[],  # Future: Insert tool functions here
        verbose=False
    )


def create_reviewer_agents(num_reviewers: int = 3) -> List[CodeAgent]:
    return [
        CodeAgent(
            name=f"Reviewer_{i+1}",
            model=llm,
            tools=[],  # Future: Add fact-checking or retrieval tools
            verbose=False
        )
        for i in range(num_reviewers)
    ]


def create_meta_reviewer_agent() -> CodeAgent:
    return CodeAgent(
        name="MetaReviewer",
        model=llm,
        tools=[],
        verbose=False
    )
