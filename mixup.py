"""Three-source-domain Mixup used by CC."""

import numpy as np
import torch


def mix_three_domains(concatenated_inputs, domain_batch_size, device):
    """Mix corresponding samples from three source batches with one weight triplet."""
    weights = np.random.dirichlet(alpha=(1.0, 1.0, 1.0), size=1)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    source_batches = [
        concatenated_inputs.narrow(0, index * domain_batch_size, domain_batch_size)
        for index in range(3)
    ]
    mixed = sum(weights[0, index] * source_batches[index] for index in range(3))
    return mixed, weights
