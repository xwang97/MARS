from huggingface_hub.inference._generated.types import ChatCompletionOutputMessage, ChatCompletionOutputToolCall
from sagemaker.predictor import retrieve_default
import uuid


def build_sagemaker_model(endpoint):
    predictor = retrieve_default(endpoint)

    def custom_model(messages, stop_sequences=None, grammar=None, **kwargs):
        payload = {
            "messages": messages,
            "max_tokens": 1024
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            # Call your SageMaker endpoint
            response = predictor.predict(payload)
            choice = response["choices"][0]["message"]

            # Try extracting content and tool_calls
            content = choice.get("content")
            tool_calls_raw = choice.get("tool_calls", [])

            tool_calls = []
            for i, raw in enumerate(tool_calls_raw):
                # Fallbacks to ensure everything is present
                call_id = raw.get("id") or f"tool_call_{uuid.uuid4().hex[:8]}"
                function = raw.get("function", {})
                name = function.get("name", f"unnamed_function_{i}")
                args = function.get("arguments", "{}")

                # Ensure valid ToolCall object
                tool_calls.append(
                    ChatCompletionOutputToolCall(
                        id=call_id,
                        type=raw.get("type", "function"),
                        function={"name": name, "arguments": args}
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

            return ChatCompletionOutputMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls if tool_calls else None
            )

        except Exception as e:
            # Fallback if anything breaks
            return ChatCompletionOutputMessage(
                role="assistant",
                content=f"[Model error] {str(e)}",
                tool_calls=None
            )
    return custom_model
