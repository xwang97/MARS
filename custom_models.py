# from huggingface_hub.inference._generated.types import ChatCompletionOutputMessage, ChatCompletionOutputToolCall
from smolagents.models import Model, ChatMessage, ChatMessageToolCall, ChatMessageToolCallDefinition
from sagemaker.predictor import retrieve_default
import uuid


class SagemakerModel(Model):
    def __init__(self, endpoint):
        self.predictor = retrieve_default(endpoint)

    def generate(messages, stop_sequences=None):
        payload = {
            "messages": messages,
            "max_tokens": 4096
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            # Call your SageMaker endpoint
            response = self.predictor.predict(payload)
            choice = response["choices"][0]["message"]

            # Try extracting content and tool_calls
            content = choice.get("content")
            tool_calls_raw = choice.get("tool_calls", [])

            tool_calls = []
            for i, tc in enumerate(tool_calls_raw):
                # Ensure valid ToolCall object
                tool_calls.append(
                    ChatMessageToolCall(
                        function=ChatMessageToolCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                    )
                )

            # If tool calls are present, suppress content to null (as expected by smolagents)
            # if tool_calls:
            #     content = None

            # Truncate on stop sequences only if content is str
            if isinstance(content, str) and stop_sequences:
                for stop in stop_sequences:
                    if stop in content:
                        content = content.split(stop)[0]

            return ChatMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )

        except Exception as e:
            # Fallback if anything breaks
            return ChatMessage(
                role="assistant",
                content=f"[Model error] {str(e)}",
                tool_calls=None
            )

