import torch

from scmolt.trackc import GaussianPosterior, gaussian_poe


def test_prior_inclusive_product_of_experts() -> None:
    first = GaussianPosterior(torch.tensor([[2.0]]), torch.zeros(1, 1))
    second = GaussianPosterior(torch.tensor([[4.0]]), torch.zeros(1, 1))
    result = gaussian_poe([first, second], include_standard_prior=True)
    assert torch.allclose(result.mean, torch.tensor([[2.0]]))
    assert torch.allclose(result.logvar.exp(), torch.tensor([[1.0 / 3.0]]))


def test_masked_expert_contributes_zero_precision() -> None:
    active = GaussianPosterior(torch.tensor([[3.0]]), torch.zeros(1, 1))
    inactive = GaussianPosterior(torch.tensor([[100.0]]), torch.zeros(1, 1))
    result = gaussian_poe(
        [active, inactive],
        masks=[torch.tensor([True]), torch.tensor([False])],
        include_standard_prior=True,
    )
    assert torch.allclose(result.mean, torch.tensor([[1.5]]))
    assert torch.allclose(result.logvar.exp(), torch.tensor([[0.5]]))
