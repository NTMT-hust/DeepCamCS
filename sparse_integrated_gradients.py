import torch

class SparseIntegratedGradients:

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

        grads_acc = torch.zeros_like(x)

        for i in range(1, self.steps + 1):

            alpha = i / self.steps

            x_step = baseline + alpha * (x - baseline)
            x_step = x_step.clone().detach().requires_grad_(True)

            out = self.model(x_step)

            prob = torch.softmax(out, dim=1)
            score = prob[0, target_class]

            grad = torch.autograd.grad(score, x_step)[0]

            grad_sparse = torch.zeros_like(grad)
            grad_sparse[sparse_mask] = grad[sparse_mask]

            grads_acc += grad_sparse

        grads_acc /= self.steps

        attr = (x - baseline) * grads_acc

        return attr
