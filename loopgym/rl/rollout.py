"""Rollout backends: completions plus the sampler's per-token log-probs.

sampler_logp is the log-prob of each sampled token under the temperature-scaled
sampling distribution (transformers applies the temperature warper before returning
`scores`, so log_softmax(scores)[token] is exactly that). The trainer recomputes its
own log-probs; the known check is that they agree within 1e-4 at temperature 1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class Completion:
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    sampler_logp: list[float] = field(default_factory=list)


class RolloutBackend(Protocol):
    def generate(
        self,
        prompts: list[list[dict]],
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[Completion]]:
        """n completions per prompt (chat messages). Rows are independent samples."""
        ...


class HFBackend:
    """Batched HuggingFace `generate` with the tokenizer chat template and left padding.

    temperature <= 0 means do_sample=False (greedy); then the n copies of a prompt are
    identical. Rows are generated in chunks of `micro_batch` so the per-step scores
    ([steps, chunk, vocab] fp32) stay small; each chunk seeds the global torch RNG
    (seed + 7919 * chunk) so every row still gets an independent, reproducible sample.
    Position ids are per-row (left padding is masked), so chunking never changes a
    row's logits."""

    def __init__(self, model, tokenizer, micro_batch: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.micro_batch = max(1, int(micro_batch))

    def generate(
        self,
        prompts: list[list[dict]],
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[Completion]]:
        prompt_ids = [
            self.tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True).input_ids for p in prompts
        ]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer has no pad token")
        rows = [ids for ids in prompt_ids for _ in range(n)]
        dev = next(self.model.parameters()).device
        do_sample = temperature > 0
        eos_id = self.tokenizer.eos_token_id
        results: list[list[Completion]] = [[] for _ in prompts]
        for c0 in range(0, len(rows), self.micro_batch):
            sub = rows[c0:c0 + self.micro_batch]
            width = max(len(r) for r in sub)
            input_ids = torch.tensor(
                [[pad_id] * (width - len(r)) + r for r in sub], dtype=torch.long).to(dev)
            attention_mask = torch.tensor(
                [[0] * (width - len(r)) + [1] * len(r) for r in sub], dtype=torch.long).to(dev)
            torch.manual_seed(seed + 7919 * (c0 // self.micro_batch))
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    # the instruct model's generation_config.json ships top_k=20 and
                    # repetition_penalty=1.05; rollouts must be exactly temperature + top_p
                    top_k=0,
                    repetition_penalty=1.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            scores = [s.float() for s in out.scores]  # [steps, chunk, V], fp32, temperature-scaled
            seqs = out.sequences
            for j, r in enumerate(sub):
                i = c0 + j
                prompt = prompt_ids[i // n]
                comp = seqs[j, width:].tolist()
                # rows that hit EOS are frozen; HF fills the rest of the sequence with pad, so trim there
                if eos_id is not None and eos_id in comp:
                    comp = comp[: comp.index(eos_id) + 1]
                logp = [
                    float(torch.log_softmax(scores[s], dim=-1)[j, token]) for s, token in enumerate(comp)
                ]
                results[i // n].append(
                    Completion(
                        prompt_ids=list(prompt),
                        completion_ids=comp,
                        text=self.tokenizer.decode(comp),
                        sampler_logp=logp,
                    )
                )
            del scores, out
        return results
