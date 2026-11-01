"""LLM policy for the desktop RL environment."""

from __future__ import annotations

from typing import Any

import torch

SYSTEM_PROMPT = """You control a Linux desktop app through the accessibility tree.
Read the tree below and decide the next action.

Line format: [k] role "name" text="..." (actions) {states}
- k = short id, role = node type, name = displayed label
- text = current content (textfields only)
- actions = available commands (click,type,sel,...)
- states = focused, selected, offscreen, etc.

Menus: [k] menu "File" (+N items, use open_menu)
Use open_menu k to see a menu's contents (clicks it, the app shows it).
Use close_menu k to collapse a menu (presses escape).

End your reply with exactly one ACTION line:
ACTION: {"op": "click", "id": <k>}
ACTION: {"op": "type", "id": <k>, "text": "..."}
ACTION: {"op": "set_value", "id": <k>, "value": "..."}
ACTION: {"op": "press", "keys": "ctrl+s"}
ACTION: {"op": "open_menu", "id": <k>}
ACTION: {"op": "close_menu", "id": <k>}
ACTION: {"op": "done"}

You may include brief reasoning before the ACTION line."""


def _build_prompt(task_instruction: str, history: list[tuple[str, str]],
                  current_tree: str) -> str:
    """Build a chat-style prompt for the model."""
    lines = [f"Task: {task_instruction}", ""]
    if history:
        lines.append("Previous actions:")
        for act, result in history[-4:]:
            lines.append(f"  Action: {act}")
            lines.append(f"  Result: {result}")
        lines.append("")
    lines.append("Current tree:")
    lines.append(current_tree)
    return "\n".join(lines)


class LMPolicy:
    """An LLM that generates actions from observations.

    Parameters
    ----------
    model:
        HuggingFace causal LM (e.g. Qwen2ForCausalLM).
    tokenizer:
        HuggingFace tokenizer (or compatible interface).
    max_new_tokens:
        Max tokens to generate per action.
    temperature:
        Sampling temperature.
    device:
        Device to run on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        max_new_tokens: int = 96,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def act(
        self,
        obs: str,
        history: list[tuple[str, str]] | None = None,
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Generate an action from the observation.

        Returns (action_text, prompt_ids, completion_ids).
        completion_ids are the generated tokens (for logprob computation in GRPO).
        """
        # The task instruction is the first line (Task: ...)
        task_line = obs.splitlines()[0] if obs else ""
        inst = task_line.replace("Task: ", "", 1) if task_line.startswith("Task: ") else obs

        # Find the current tree part (everything after "Current tree:" or the rendered tree)
        tree_start = obs.find("\n")
        if "Last action:" in obs:
            tree_end = obs.rfind("Last action:")
            current_tree = obs[tree_start:tree_end].strip()
        else:
            current_tree = obs[tree_start:].strip()

        history = history or []
        prompt_text = _build_prompt(inst, history, current_tree)

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt_text}\n\nAssistant:"

        # Tokenize
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=False,
        )

        prompt_ids = inputs["input_ids"][0]
        completion_ids = out[0, prompt_len:]
        action_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return action_text.strip(), prompt_ids, completion_ids
