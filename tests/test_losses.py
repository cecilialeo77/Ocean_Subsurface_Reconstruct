import torch

from glorys_swin_reconstruction.losses import (
    depth_balanced_masked_mse_loss,
    vertical_gradient_transport_loss,
)


def test_depth_balanced_loss_weights_depths_equally():
    prediction = torch.tensor([[[[2.0, 2.0]], [[3.0, 0.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 1.0]], [[1.0, 0.0]]]])
    # Depth losses are 4 and 9, regardless of their different valid counts.
    loss = depth_balanced_masked_mse_loss(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor(6.5))


def test_empty_mask_has_differentiable_zero_loss():
    prediction = torch.randn(1, 2, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction)
    loss = depth_balanced_masked_mse_loss(prediction, target, mask)
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
