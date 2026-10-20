"""Build the chat string on the host. The simulator tokenizes, runs the model, and detokenizes."""

from pathlib import Path

from serve.llama import Llama


class Engine:
    def __init__(self, model_dir):
        self.model = Llama(model_dir)
        self.eos_id = self.model.eos_id

    def chat_prompt(self, messages):
        # SmolLM2's chat template inserts this system turn when the caller did not.
        lines = []
        if not messages or messages[0].get("role") != "system":
            lines.append(
                "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            )
        for message in messages:
            lines.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        lines.append("<|im_start|>assistant\n")
        return "".join(lines)

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
        text, produced, prompt_tokens = self.model.complete(prompt, max_tokens, self.eos_id, temperature, seed)
        return {
            "text": text,
            "token_ids": produced,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(produced),
        }

    def close(self):
        self.model.close()
