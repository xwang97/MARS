from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel, DuckDuckGoSearchTool, VisitWebpageTool
from typing import List
import random
from utils import get_openai_api_key


# Adjust the logging level for smolagents
verbosity = -1

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
openai_api_key = get_openai_api_key("../openai_api_key_4_XiaoWang.txt")
gpt_4o_mini = OpenAIServerModel(model_id="gpt-3.5-turbo", api_base="https://api.openai.com/v1", api_key=openai_api_key)
author_llm = gpt_4o_mini
reviewer_llms = [
    gpt_4o_mini
]
meta_llm = gpt_4o_mini

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
            max_steps=1,
        )
    else:
        return ToolCallingAgent(
            name="Author",
            model=author_llm,
            tools=[],  # Future: Insert tool functions here
            verbosity_level=verbosity,
            max_steps=1,
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
                tools=[],
                verbosity_level=verbosity,
                max_steps=1,
            )
        else:
            reviewer = ToolCallingAgent(
                name=f"Reviewer_{i+1}",
                model=llm,
                tools=[],
                verbosity_level=verbosity,
                max_steps=1,
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
            max_steps=1,
        )
    else:
        return ToolCallingAgent(
            name="MetaReviewer",
            model=meta_llm,
            tools=[],
            verbosity_level=verbosity,
            max_steps=1,
        )


# ======= Definition of the AWS agent class ================
class AWSAgent:
    def __init__(self, name, endpoint_name, model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=4096):
        self.name = name
        self.endpoint_name = endpoint_name
        self.model = model
        self.predictor = retrieve_default(endpoint_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_new_tokens = max_new_tokens
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_aws(messages)["content"]  # for compatibility
        elif isinstance(prompt, list):
            return self._call_aws(prompt)  # chat-style list[dict]

    def _call_aws(self, messages: list[dict]) -> dict:
        formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        payload = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "details": True,
                "decoder_input_details": True
            }
        }
        response = self.predictor.predict(payload)

        input_tokens = len(response['details']['prefill'])
        output_tokens = response["details"]["generated_tokens"]
        self.total_tokens += (input_tokens + output_tokens)
        self.token_log.append({
            "agent": self.name,
            "model": self.model,
            "endpoint": self.endpoint_name,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": self.total_tokens,
            "messages_preview": messages[-1]["content"][:100]
        })
        return {
            "role": "assistant",
            "content": response["generated_text"].strip()
        }


# ======= Definition of the agent class using HuggingFace ================
class HFAgent:
    def __init__(self, name, model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=4096):
        self.name = name
        self.model = model
        self.client = InferenceClient(api_key=hf_api_key)
        self.max_new_tokens = max_new_tokens
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_hf(messages)["content"]  # Return string for backward compatibility
        elif isinstance(prompt, list):
            return self._call_hf(prompt)  # Return dict for debate/chat-style use

    def _call_hf(self, messages: list[dict]) -> dict:
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