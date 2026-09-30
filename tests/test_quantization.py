"""Vector quantiser, codebook lookup and utilisation statistics."""
import numpy as np
import pytest
import torch

from medical_tokenizer.quantization.stats import CodeUsageTracker, usage_stats
from medical_tokenizer.quantization.vq import IdentityQuantizer, VectorQuantizer, build_quantizer


def test_nearest_code_selection_matches_bruteforce():
    torch.manual_seed(0)
    vq = VectorQuantizer(16, 4, normalize=False, init="random").eval()
    z = torch.randn(5, 7, 4)
    idx = vq.codes(z)
    brute = torch.cdist(z.reshape(-1, 4), vq.codebook).argmin(1).view(5, 7)
    assert torch.equal(idx, brute)
    out = vq(z)
    assert torch.equal(out.indices, idx)
    torch.testing.assert_close(out.z_q, vq.lookup(idx))            # forward value == decoded codes


def test_cosine_codebook_normalises():
    vq = VectorQuantizer(8, 4, normalize=True, init="random").eval()
    z = torch.randn(10, 4) * 7
    assert torch.equal(vq.codes(z), vq.codes(z / 7))              # scale invariant


def test_straight_through_gradient_reaches_encoder():
    torch.manual_seed(0)
    vq = VectorQuantizer(8, 3, update="loss", normalize=False, init="random")
    z = torch.randn(20, 3, requires_grad=True)
    out = vq(z)
    (out.z_q.sum() + out.loss).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert vq.codebook.grad is not None and vq.codebook.grad.abs().sum() > 0     # codebook loss trains codes


def test_losses_match_definitions():
    torch.manual_seed(1)
    vq = VectorQuantizer(8, 3, update="loss", beta=0.25, normalize=False, init="random", restart_dead=False)
    z = torch.randn(30, 3)
    out = vq(z)
    e = vq.lookup(out.indices)
    mse = ((z - e) ** 2).mean()
    assert out.commitment_loss.item() == pytest.approx(mse.item(), rel=1e-5)
    assert out.codebook_loss.item() == pytest.approx(mse.item(), rel=1e-5)
    assert out.loss.item() == pytest.approx(1.25 * mse.item(), rel=1e-5)


def test_ema_mode_moves_codes_without_codebook_loss():
    torch.manual_seed(0)
    vq = VectorQuantizer(4, 2, update="ema", decay=0.5, normalize=False, restart_dead=False, init="random")
    before = vq.codebook.detach().clone()
    out = vq(torch.randn(64, 2) + 3)
    assert out.codebook_loss.item() == 0.0
    assert not torch.allclose(before, vq.codebook)
    assert not vq.codebook.requires_grad


def test_dead_code_restart_revives_codes():
    torch.manual_seed(0)
    vq = VectorQuantizer(32, 2, update="ema", normalize=False, restart_dead=True, restart_threshold=0.5,
                         init="random", decay=0.5)
    z = torch.randn(256, 2) * 0.01 + 5                            # all inputs near one point
    restarted = sum(vq(z).restarted for _ in range(5))
    assert restarted > 0
    d = torch.cdist(vq.codebook.detach(), z).min(1).values
    assert (d < 0.1).float().mean() > 0.5                        # most codes moved onto the data


def test_data_init_initialises_from_first_batch():
    vq = VectorQuantizer(8, 2, normalize=False, init="data")
    z = torch.randn(100, 2) + 10
    vq(z)
    assert bool(vq.initialised) and vq.codebook.detach().mean() > 5


def test_identity_quantizer_and_builder():
    q = build_quantizer({"type": "none"}, 4)
    assert isinstance(q, IdentityQuantizer)
    z = torch.randn(3, 4)
    out = q(z)
    assert out.indices is None and torch.equal(out.z_q, z)
    v = build_quantizer({"type": "vq", "n_codes": 16}, 4)
    assert isinstance(v, VectorQuantizer) and v.n_codes == 16
    with pytest.raises(ValueError):
        build_quantizer({"type": "bogus"}, 4)


def test_usage_stats_uniform_and_collapsed():
    u = usage_stats(np.full(8, 10))
    assert u["active_codes"] == 8 and u["utilization"] == 1.0 and u["dead_codes"] == 0
    assert u["entropy_bits"] == pytest.approx(3.0) and u["perplexity"] == pytest.approx(8.0)
    assert u["norm_entropy"] == pytest.approx(1.0)
    c = np.zeros(8, int)
    c[2] = 50
    u = usage_stats(c)
    assert u["active_codes"] == 1 and u["dead_pct"] == pytest.approx(87.5) and u["entropy_bits"] == 0.0
    assert u["perplexity"] == pytest.approx(1.0) and u["top10_share"] == 1.0
    assert usage_stats(np.zeros(4, int))["active_codes"] == 0


def test_tracker_accumulates():
    t = CodeUsageTracker(4)
    t.update(torch.tensor([0, 1, 1]))
    t.update(np.array([[3, 3]]))
    assert t.counts.tolist() == [1, 2, 0, 2]
    assert t.stats()["active_codes"] == 3
