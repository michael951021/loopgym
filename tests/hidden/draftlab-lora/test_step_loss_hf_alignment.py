"""Hidden check, draftlab-lora.

step_loss must be next-token cross-entropy with HF alignment: logits row t
predicts ids[t+1] (== model(ids, labels=ids).loss), padded tail ignored.
Bad code scores the token-echoer (logits[t] vs ids[t]), a copycat whose
train loss -> 0.007 on data where real next-token loss stays ~2. Known
answer: HF's own labels-loss, on the module's own tiny base. Seeded, CPU.
"""
import pytest
import torch

from draftlab.drafters.lora import build_base, step_loss


@pytest.fixture(scope="module")
def model():
    return build_base(None, True, False, 0).eval()


def test_step_loss_matches_hf_labels_loss(model):
    torch.manual_seed(0)
    chunk = torch.randint(0, 200, (64,))
    ids = chunk.unsqueeze(0)
    loss = step_loss(model, chunk, 64)
    with torch.no_grad():
        ref = model(input_ids=ids, labels=ids).loss
    assert abs(loss.item() - ref.item()) < 1e-5, (
        f"step_loss {loss.item():.6f} != HF labels-loss {ref.item():.6f}; "
        "alignment must be next-token (logits[:-1] vs ids[1:])"
    )


def test_padding_tail_ignored(model):
    torch.manual_seed(1)
    chunk = torch.randint(0, 200, (64,))
    n = 47
    loss = step_loss(model, chunk, n)
    with torch.no_grad():
        ref = model(input_ids=chunk[:n].unsqueeze(0),
                    labels=chunk[:n].unsqueeze(0)).loss
    assert abs(loss.item() - ref.item()) < 1e-5, (
        f"padded chunk {loss.item():.6f} != valid prefix {ref.item():.6f}; "
        "positions past n_valid must be ignored (labels -100)"
    )
