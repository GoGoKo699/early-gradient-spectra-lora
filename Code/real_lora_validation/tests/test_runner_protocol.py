import sys
import types

import torch

# The loader protocol itself only needs torch.  Supply import stubs when the
# packaging/test host does not have the full Hugging Face stack installed.
try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    sys.modules["transformers"] = transformers_stub

from run_real_lora_validation import make_loader


def flatten_batches(loader):
    return torch.cat([batch[0].reshape(-1) for batch in loader])


def test_fresh_loaders_reproduce_identical_batch_order():
    blocks = torch.arange(80, dtype=torch.long).reshape(20, 4)
    first = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    second = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    assert torch.equal(flatten_batches(first), flatten_batches(second))


def test_consumed_loader_does_not_change_new_loader_order():
    blocks = torch.arange(80, dtype=torch.long).reshape(20, 4)
    consumed = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    _ = list(consumed)
    fresh_a = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    fresh_b = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    assert torch.equal(flatten_batches(fresh_a), flatten_batches(fresh_b))
