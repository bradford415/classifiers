import torch


class Unnormalize:
    """Unormalize a tensor that normalized by torchvision.transforms.Normalize

    Normalize subtracts mean and divides by std dev so to Unnormalize we need to
    multiply by the std dev and add the mean
    """

    def __init__(self, mean: list[float], std: list[float], inplace=False):
        """Initialize the unnormalize class

        Args:
            mean: the mean of the dataset for each channel
            std: the standard deviation of the dataset for each channel
        """
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        """
        This code is largely based on: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L905
        Args:
            tensor: Tensor image of size (B, C, H, W) or (C, H, W) to be unnormalized.
        Returns:
            Tensor: Unormalized image.
        """

        if tensor.ndim < 3:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
            )

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)

        if not self.inplace:
            tensor = tensor.clone()

        # change shape to broadcast
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        # Modifies in place `_`
        tensor.mul_(std).add_(mean)
        # The normalize code -> t.sub_(m).div_(s)

        return tensor
