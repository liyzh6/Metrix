"""Loss terms specific to class-conditional consistency (CC)."""

import torch
from torch.nn import functional as F


def class_conditional_domain_loss(features, labels, domains):
    """Align source-domain centers separately for every class in the batch."""
    total = features.new_tensor(0.0)
    valid_classes = 0

    for label in labels.unique():
        class_mask = labels == label
        centers = []
        for domain in domains[class_mask].unique():
            domain_mask = class_mask & (domains == domain)
            centers.append(features[domain_mask].mean(dim=0))
        if len(centers) >= 2:
            centers = torch.stack(centers)
            shared_center = centers.mean(dim=0, keepdim=True)
            total = total + (centers - shared_center).square().mean()
            valid_classes += 1

    return total / max(valid_classes, 1)


def compact_separation_loss(features, labels, margin=2.0):
    """Make samples compact within a class and separate different class centers."""
    centers = []
    compactness = features.new_tensor(0.0)

    for label in labels.unique():
        group = features[labels == label]
        center = group.mean(dim=0)
        centers.append(center)
        compactness = compactness + (group - center).square().sum(dim=1).mean()

    compactness = compactness / max(len(centers), 1)
    if len(centers) < 2:
        return compactness

    centers = torch.stack(centers)
    distances = torch.cdist(centers, centers, p=2)
    off_diagonal = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
    separation = F.relu(margin - distances[off_diagonal]).square().mean()
    return compactness + separation
