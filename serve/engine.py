"""Build the chat string on the host from the checkpoint's chat template.

The simulator tokenizes, runs the model, and detokenizes. Streaming detokenizes
with the host ``tokenizers`` package and does not call back into the model.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

from serve.llama import Llama

_FOR = "{% for message in messages %}"
_INJECT_HEAD = "{% if loop.first and messages[0]['role'] != 'system' %}{{ '"
_INJECT_TAIL = "' }}{% endif %}"
_PIECE = "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
_TAIL = "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

MAX_NEW_TOKENS = 48


class ChatSkeleton:
    """The one template shape both SmolLM checkpoints use. ``system`` is None when the file has no inject."""

    def __init__(self, system):
        self.system = system


def parse_chat_template(template):
    if not isinstance(template, str) or not template.startswith(_FOR):
        raise ValueError("unsupported chat template")
    rest = template[len(_FOR) :]
    system = None
    if rest.startswith(_INJECT_HEAD):
        end = rest.find(_INJECT_TAIL)
        if end < 0:
            raise ValueError("unsupported chat template")
        system = rest[len(_INJECT_HEAD) : end]
        rest = rest[end + len(_INJECT_TAIL) :]
    if not rest.startswith(_PIECE):
        raise ValueError("unsupported chat template")
    rest = rest[len(_PIECE) :]
    if rest != _TAIL:
        raise ValueError("unsupported chat template")
    return ChatSkeleton(system)


def render_chat(skeleton, messages, add_generation_prompt=True):
    """One pass over messages. The system line is inside the loop, so an empty list does not inject it."""
    if not isinstance(skeleton, ChatSkeleton):
        skeleton = parse_chat_template(skeleton)
    parts = []
    for index, message in enumerate(messages):
        if index == 0 and skeleton.system is not None and message.get("role") != "system":
            parts.append(skeleton.system)
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def stable_text(tokenizer, token_ids):
    """Decode with the host tokenizer. Hold back one trailing U+FFFD a later token might complete."""
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    if text.endswith("\ufffd"):
        return text[:-1]
    return text


class Engine:
    def __init__(self, model_dir, context=None):
        model_dir = Path(model_dir)
        self.model = Llama(model_dir, context=context)
        self.eos_id = self.model.eos_id
        with open(model_dir / "tokenizer_config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        self.skeleton = parse_chat_template(config.get("chat_template"))
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._stream_ids = []

    def chat_prompt(self, messages):
        return render_chat(self.skeleton, messages, add_generation_prompt=True)

    def start_stream(self):
        self._stream_ids = []

    def take_token(self, token_id):
        """Host-only. Safe to call from the generate callback: it does not touch the model lock."""
        self._stream_ids.append(int(token_id))
        return stable_text(self._tokenizer, self._stream_ids)

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0, top_k=0, on_token=None):
        limit = int(max_tokens)
        if limit > MAX_NEW_TOKENS:
            raise ValueError("max_tokens must be at most 48")
        text, produced, prompt_tokens = self.model.complete(
            prompt, limit, self.eos_id, temperature, seed, top_k, on_token
        )
        return {
            "text": text,
            "token_ids": produced,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(produced),
        }

    def close(self):
        self.model.close()
