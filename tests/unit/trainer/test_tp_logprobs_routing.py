"""The TP log-prob patch must leave every model other than the TP policy on the plain path."""

import types

import grail.trainer.algorithms.grpo as grpo_module
import grail.trainer.distributed.logprobs as logprobs_module
from grail.trainer.distributed.training_service import DistributedTrainingService


def test_reference_model_bypasses_tp_logprobs(monkeypatch) -> None:
    calls = []
    tp_calls = []

    def plain(model, *args, **kwargs):
        calls.append(model)
        return "plain"

    def distributed(model, *args, **kwargs):
        tp_calls.append(model)
        return "tp", None, None

    monkeypatch.setattr(grpo_module, "compute_logprobs", plain)
    monkeypatch.setattr(logprobs_module, "compute_logprobs_distributed", distributed)
    svc = DistributedTrainingService.__new__(
        DistributedTrainingService
    )  # no __init__: only what the patch reads
    svc._tp_group = object()
    svc.tp_size = 2
    svc.train_model = object()
    svc.train_config = types.SimpleNamespace(logit_chunk_size=4)
    originals = svc._patch_logprobs_for_tp()
    try:
        ref_model = object()
        assert grpo_module.compute_logprobs(ref_model, None, None, [], []) == "plain"
        assert calls == [ref_model]
        # the policy itself still takes the TP path
        assert grpo_module.compute_logprobs(svc.train_model, None, None, [], []) == "tp"
        assert tp_calls == [svc.train_model] and calls == [ref_model]
    finally:
        svc._restore_logprobs(originals)
