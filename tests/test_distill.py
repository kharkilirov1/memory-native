"""Phase 4 witness (toy scale): a distillation finetune pulls the degraded counter student back
toward its fp teacher -- the KL to the teacher drops. Runs on CPU with a tiny random Qwen2, so it
proves the *mechanism* (network-level recovery via distill), not a production recovery curve."""
import copy
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM

from memory_native.donor.qwen import qwen_to_counter
from memory_native.eval import perplexity
from memory_native.recovery import ResidentTeacher, distill_finetune, kd_divergence


def _tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg).eval()


def _mean_kl(student, teacher_model, batches):
    was_training = student.training
    student.eval()
    total = 0.0
    with torch.no_grad():
        for ids in batches:
            total += float(kd_divergence(student(ids).logits, teacher_model(ids).logits, 1.0))
    student.train(was_training)
    return total / len(batches)


def _overfit(model, batch, *, steps=80, lr=1e-2):
    """Make the fp teacher *confident* on a fixed batch (sharp next-token distributions), so that
    ternarizing it into the counter format is a real, KL-visible degradation to recover from --
    unlike a random model, whose softmax is near-uniform and barely moves under ternarization."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch, labels=batch)
        out.loss.backward()
        opt.step()
    return model.eval()


def test_distill_reduces_kl_to_teacher():
    fp = _tiny(0)
    torch.manual_seed(1)
    batch = torch.randint(0, 64, (2, 8))
    _overfit(fp, batch, steps=80, lr=1e-2)       # sharp teacher -> ternarization really hurts
    teacher_model = copy.deepcopy(fp).eval()

    student = fp                       # in-place swap: student becomes the degraded counter model
    qwen_to_counter(student)
    batches = [batch]

    before = _mean_kl(student, teacher_model, batches)
    distill_finetune(student, ResidentTeacher(teacher_model), batches,
                     steps=150, temperature=2.0, lr=5e-3)
    after = _mean_kl(student, teacher_model, batches)

    print(f"\n[distill witness] KL(teacher||student): before={before:.4f} after={after:.4f} "
          f"({100*(before-after)/before:.1f}% closer)")
    assert after < before


def test_perplexity_runs_and_restores_mode():
    m = _tiny(0)
    m.train()
    batches = [torch.randint(0, 64, (2, 8)) for _ in range(2)]
    ppl = perplexity(m, batches)
    assert math.isfinite(ppl) and ppl > 0
    assert m.training is True          # mode restored after measurement
