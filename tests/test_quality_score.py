import types
from types import SimpleNamespace

import numpy as np
import torch

from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        # very small vocab for tests
        self._tok2id = {"<pad>": 0, "<eos>": 1}

    def _encode_text(self, s: str) -> list[int]:
        # map each visible ascii char to an id > 1, stable and deterministic
        ids = [ord(c) % 97 + 2 for c in s]
        return ids + [self.eos_token_id]

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=4096):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_text(t) for t in texts]
        maxlen = max(len(x) for x in encoded)
        maxlen = min(maxlen, max_length)
        # left-truncate if beyond max_length
        proc = []
        for seq in encoded:
            if len(seq) > maxlen:
                seq = seq[-maxlen:]
            pad = [self.pad_token_id] * (maxlen - len(seq))
            proc.append(pad + seq)
        input_ids = torch.tensor(proc, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, ids_batch, skip_special_tokens=True):
        # reverse of _encode_text (approximate)
        out = []
        for ids in ids_batch:
            ids = [int(x) for x in ids if int(x) not in (self.pad_token_id,)]
            # drop trailing eos if present
            if ids and ids[-1] == self.eos_token_id:
                ids = ids[:-1]
            out.append("".join(chr((i - 2) % 97 + 97) for i in ids))
        return out

    def decode(self, ids, skip_special_tokens=True):
        return self.batch_decode(ids.unsqueeze(0))[0]


def _make_worker_with_dummy_tokenizer():
    # Bypass heavy ActorRolloutRefWorker.__init__
    worker = object.__new__(ActorRolloutRefWorker)
    worker.model_config = SimpleNamespace(tokenizer=DummyTokenizer())
    return worker


def test_build_quality_score_prompts_from_raw_prompt():
    worker = _make_worker_with_dummy_tokenizer()

    # raw_prompt as list[dict]
    raw_prompt = np.array([
        {"input": "1+1?", "output": "2"},
        {"input": "abc", "output": "xyz"},
    ], dtype=object)

    dp = DataProto(
        batch=None,
        non_tensor_batch={"raw_prompt": raw_prompt},
        meta_info={},
    )

    out = worker._build_quality_score_prompts(dp)

    # should be TensorDict and support .to()
    assert isinstance(out.batch, TensorDict)
    assert "input_ids" in out.batch and "attention_mask" in out.batch
    assert out.batch.batch_size[0] == 2

    # to(device) should not raise
    out2 = out.to("cpu")
    assert out2.batch.device.type in ("cpu",)


def test_extract_quality_scores_from_generation_box_format():
    worker = _make_worker_with_dummy_tokenizer()

    # Build responses that decode to include a box score like "\\box{8.0}"
    # We directly bypass decoding by constructing a text then re-encoding with DummyTokenizer
    texts = ["score \\box{8.0}", "score \\box{5}"]
    tok = worker.model_config.tokenizer
    enc = tok(texts, return_tensors="pt")

    # Use responses as encoded sequences
    td = TensorDict({"responses": enc["input_ids"]}, batch_size=(len(texts),))
    dp = DataProto(batch=td, non_tensor_batch={})

    scores = worker._extract_quality_scores_from_generation(dp)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (2,)
    # normalized by /10.0
    assert torch.isclose(scores[0], torch.tensor(0.8), atol=1e-3)
    assert torch.isclose(scores[1], torch.tensor(0.5), atol=1e-3)


