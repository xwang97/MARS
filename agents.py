from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, DuckDuckGoSearchTool, VisitWebpageTool
from custom_models import build_sagemaker_model
from typing import List
import random


# Adjust the logging level for smolagents
verbosity = 1

###############################################################################
# Uncomment this part if you want to use HuggingFace API for backend models
# # LLM setup
# author_llm = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
# # Qwen/Qwen2.5-Coder-7B-Instruct, nebius
# # deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct, nebius
# reviewer_llms = [
#     HfApiModel(model_id="meta-llama/Llama-3.1-8B-Instruct"),
#     HfApiModel(model_id="meta-llama/Llama-3.2-3B-Instruct", provider="together"),
#     HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct"),
#     HfApiModel(model_id="mistralai/Mistral-7B-Instruct-v0.3"),
#     HfApiModel(model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503", provider="nebius"),
#     HfApiModel(model_id="Qwen/Qwen2.5-7B-Instruct", provider="together"),
#     HfApiModel(model_id="Qwen/Qwen2.5-1.5B-Instruct", provider="together"),
#     HfApiModel(model_id="Qwen/Qwen2.5-32B-Instruct", provider="nebius") 
# ]
# meta_llm = HfApiModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096)
###############################################################################

# ========== LLM Setup ================
endpoint_name1 = "jumpstart-dft-llama-3-1-8b-instruct-20250506-154935"
# endpoint_name2 = "jumpstart-dft-llama-codellama-7b-in-20250506-202728"
# endpoint_name3 = Qwen Coder 7B
# endpoint_name4 = DeepSeek distill or GPT
author_llm = build_sagemaker_model(endpoint_name1)
reviewer_llms = [
    build_sagemaker_model(endpoint_name1),
    # build_sagemaker_model(endpoint_name2)
]
meta_llm = build_sagemaker_model(endpoint_name1)

reviewer_styles = [
    "Be conservative. Flag anything you're unsure about. Avoid giving the benefit of the doubt.",
    "Be balanced. Use both internal consistency and known facts before making a decision.",
    "Be skeptical. Assume statements may be incorrect unless strongly supported.",
    "Be generous. Only flag as hallucinated if there is clear inconsistency or falsehood.",
    "Focus on logical coherence and internal support only. Avoid relying on world knowledge unless essential.",
    "Be thorough and critical. Use both passage and external evidence to verify claims carefully."
]

# ========== Tool Definitions ===========
web_search = DuckDuckGoSearchTool(max_results=3)
visit_webpage = VisitWebpageTool()


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
        # llm = reviewer_llms[i]
        # llm = HfApiModel(temperature=1+0.2*i)
        if agent_type == "code":
            reviewer = CodeAgent(
                name=f"Reviewer_{i+1}",
                model=llm,
                tools=[web_search, visit_webpage],
                verbosity_level=verbosity,
                max_steps=5,
            )
        else:
            reviewer = ToolCallingAgent(
                name=f"Reviewer_{i+1}",
                model=llm,
                tools=[web_search, visit_webpage],
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
