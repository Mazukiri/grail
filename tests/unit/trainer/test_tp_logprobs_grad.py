"""tp_chunked_logprobs on tensor-parallel ranks.

Every TP rank backprops the same loss, so the gradients of its lm_head shard and of the hidden
states must be those of that one loss on the full vocabulary. Both modes: without entropy (what
FSDP2 training uses) and with it.
"""

import datetime
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from grail.trainer.distributed.logprobs import tp_chunked_logprobs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, entropy: bool, port: int, errors) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=120)
    )
    try:
        torch.manual_seed(0)
        B, S, H, V = 2, 12, 16, 64
        hidden = torch.randn(B, S, H)
        w_full = torch.randn(V, H) * 3.0
        labels = torch.randint(0, V, (B, S))
        weights = torch.randn(B, S)  # per-token weights, like advantages
        rows = slice(rank * V // world, (rank + 1) * V // world)
        w = w_full[rows].clone().requires_grad_()
        h = hidden.clone().requires_grad_()
        logp, ent = tp_chunked_logprobs(
            lambda x: x @ w.T, h, labels, 4, dist.group.WORLD, return_entropy=entropy
        )
        loss = (weights * logp).sum()
        if entropy:
            loss = loss + 0.1 * ent.sum()
        loss.backward()

        wf = w_full.clone().requires_grad_()
        hf = hidden.clone().requires_grad_()
        lp = torch.log_softmax(hf @ wf.T, -1)
        ref = (weights * lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)).sum()
        if entropy:
            ref = ref - 0.1 * (lp.exp() * lp).sum()
        ref.backward()  # one loss: the one every TP rank computes
        grad_h = h.grad.clone()
        dist.all_reduce(grad_h)  # the hidden states' gradient sums the shards' contributions
        errors[rank] = max(
            ((w.grad - wf.grad[rows]).norm() / wf.grad[rows].norm()).item(),
            ((grad_h - hf.grad).norm() / hf.grad.norm()).item(),
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("world", [2, 4])
def test_tp_chunked_logprobs_gradient_matches_one_loss(world: int, entropy: bool) -> None:
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(world, entropy, _free_port(), errors), nprocs=world, join=True)
    assert max(errors.values()) < 1e-5, dict(errors)
