"""Data-independent tests for Metrix core components."""

import unittest

import torch

from classifier import Classifier
from extractor import FeatureExtractor
from loss import class_conditional_domain_loss, compact_separation_loss
from mixup import mix_three_domains


class CoreTest(unittest.TestCase):
    def test_network_shapes_and_backward(self):
        extractor = FeatureExtractor()
        classifier = Classifier(num_classes=4)
        inputs = torch.randn(6, 1, 1024)
        logits = classifier(extractor(inputs))
        self.assertEqual(tuple(logits.shape), (6, 4))
        logits.square().mean().backward()

    def test_mixup_weights(self):
        inputs = torch.randn(12, 1, 1024)
        mixed, weights = mix_three_domains(inputs, 4, torch.device("cpu"))
        self.assertEqual(tuple(mixed.shape), (4, 1, 1024))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6))

    def test_losses_are_finite_and_differentiable(self):
        features = torch.randn(12, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1] * 3)
        domains = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        loss = (
            class_conditional_domain_loss(features, labels, domains)
            + compact_separation_loss(features, labels, margin=2.0)
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
