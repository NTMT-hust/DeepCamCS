import torch

class SparseIntegratedGradients:
    """Integrated Gradients computed separately for each input channel.

    For channel c, only channel c moves along the baseline -> input path while
    the other channels stay at their real values. The result has the same shape
    as the input, (1, C, H, W), so downstream code is unchanged.
    """

    def __init__(self, model, device, steps=8):
        self.model = model
        self.device = device
        self.steps = steps

    def attribute(self, x, target_class):

        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("SparseIntegratedGradients.attribute expects one image shaped (1, C, H, W).")
        target_class = int(target_class)

        baseline = torch.zeros_like(x)
        sparse_mask = (x != 0)
        attr = torch.zeros_like(x)

        for c in range(x.shape[1]):
            grads_acc = torch.zeros_like(x[:, c:c + 1])

            for i in range(1, self.steps + 1):
                alpha = i / self.steps

                # Only channel c is scaled; the other channels keep their real values.
                x_step = x.clone()
                x_step[:, c] = baseline[:, c] + alpha * (x[:, c] - baseline[:, c])
                x_step = x_step.detach().requires_grad_(True)

                out = self.model(x_step)

                prob = torch.softmax(out, dim=1)
                score = prob[0, target_class]

                grad = torch.autograd.grad(score, x_step)[0][:, c:c + 1]

                grad_sparse = torch.zeros_like(grad)
                mask_c = sparse_mask[:, c:c + 1]
                grad_sparse[mask_c] = grad[mask_c]

                grads_acc += grad_sparse

            grads_acc /= self.steps

            attr[:, c:c + 1] = (x[:, c:c + 1] - baseline[:, c:c + 1]) * grads_acc

        return attr
