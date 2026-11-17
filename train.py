"""Self-contained training loop for the CC method."""

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from classifier import Classifier
from extractor import FeatureExtractor
from loss import class_conditional_domain_loss, compact_separation_loss
from mixup import mix_three_domains


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CCTrainer:
    def __init__(self, args, datasets, num_classes, run_dir):
        self.args = args
        self.run_dir = Path(run_dir)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor().to(self.device)
        self.classifier = Classifier(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.extractor.parameters()) + list(self.classifier.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.steps, gamma=args.gamma
        )
        self.loaders = [
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=index < 3,
                num_workers=args.num_workers,
                pin_memory=self.device.type == "cuda",
                drop_last=False,
            )
            for index, dataset in enumerate(datasets)
        ]
        self.history = []

    def _train_epoch(self):
        self.extractor.train()
        self.classifier.train()
        totals = {name: 0.0 for name in ("total", "classification", "domain", "metric", "mixup")}
        sample_count = 0

        for batches in zip(*self.loaders[:3]):
            source_inputs = [batch[0].to(self.device, non_blocking=True) for batch in batches]
            source_labels = [batch[1].to(self.device, non_blocking=True) for batch in batches]
            batch_sizes = {inputs.size(0) for inputs in source_inputs}
            if len(batch_sizes) != 1:
                raise RuntimeError("The three source batches must have equal sizes for Mixup")
            domain_batch_size = source_inputs[0].size(0)

            # Concat preserves the three domain blocks used by both CC and Mixup.
            inputs = torch.cat(source_inputs, dim=0)
            labels = torch.cat(source_labels, dim=0)
            domains = torch.cat([
                torch.full((domain_batch_size,), index, dtype=torch.long, device=self.device)
                for index in range(3)
            ])

            features = self.extractor(inputs)
            logits = self.classifier(features)
            classification = self.criterion(logits, labels)
            normalized_features = F.normalize(features, p=2, dim=1)
            domain = class_conditional_domain_loss(normalized_features, labels, domains)
            metric = compact_separation_loss(
                normalized_features, labels, margin=self.args.metric_margin
            )

            mixed_inputs, weights = mix_three_domains(inputs, domain_batch_size, self.device)
            mixed_logits = self.classifier(self.extractor(mixed_inputs))
            mixup = sum(
                weights[0, index] * self.criterion(mixed_logits, source_labels[index])
                for index in range(3)
            )
            total = (
                classification
                + self.args.lambda_domain * domain
                + self.args.lambda_metric * metric
                + self.args.lambda_mixup * mixup
            )

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            combined_size = labels.size(0)
            sample_count += combined_size
            for name, value in (
                ("total", total), ("classification", classification),
                ("domain", domain), ("metric", metric), ("mixup", mixup),
            ):
                totals[name] += float(value.item()) * combined_size

        return {name: value / sample_count for name, value in totals.items()}

    @torch.no_grad()
    def _evaluate(self):
        self.extractor.eval()
        self.classifier.eval()
        total_loss, correct, sample_count = 0.0, 0, 0
        for inputs, labels in self.loaders[3]:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.classifier(self.extractor(inputs))
            loss = self.criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            sample_count += labels.size(0)
        return total_loss / sample_count, correct / sample_count

    def _save_model(self, filename, epoch, target_accuracy):
        torch.save(
            {
                "epoch": epoch,
                "target_accuracy": target_accuracy,
                "extractor": self.extractor.state_dict(),
                "classifier": self.classifier.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "args": vars(self.args).copy(),
            },
            self.run_dir / filename,
        )

    def run(self):
        started = time.time()
        best_accuracy, best_epoch = -1.0, -1
        logging.info("device=%s", self.device)
        logging.info(
            "loss = CE + %.4g*domain + %.4g*metric + %.4g*mixup",
            self.args.lambda_domain, self.args.lambda_metric, self.args.lambda_mixup,
        )

        for epoch in range(self.args.epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            train_metrics = self._train_epoch()
            target_loss, target_accuracy = self._evaluate()
            row = {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "target_loss": target_loss,
                "target_accuracy": target_accuracy,
            }
            self.history.append(row)
            logging.info(
                "epoch=%d/%d total=%.4f CE=%.4f domain=%.6f metric=%.4f "
                "mixup=%.4f target_loss=%.4f target_acc=%.4f",
                epoch, self.args.epochs - 1, train_metrics["total"],
                train_metrics["classification"], train_metrics["domain"],
                train_metrics["metric"], train_metrics["mixup"],
                target_loss, target_accuracy,
            )
            if target_accuracy > best_accuracy:
                best_accuracy, best_epoch = target_accuracy, epoch
                # This follows the original project: target accuracy selects the best checkpoint.
                self._save_model("best_model.pt", epoch, target_accuracy)
            save_every = max(0, int(getattr(self.args, "save_every", 0)))
            if save_every and (epoch + 1) % save_every == 0:
                self._save_model(f"epoch_{epoch + 1:04d}.pt", epoch, target_accuracy)
            self.scheduler.step()

        self._save_model("last_model.pt", self.args.epochs - 1, self.history[-1]["target_accuracy"])
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

        summary = {
            "data_name": self.args.data_name,
            "task_name": self.args.task_name,
            "transfer_task": self.args.transfer_task,
            "seed": self.args.seed,
            "num_classes": self.args.num_classes,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "final_accuracy": self.history[-1]["target_accuracy"],
            "last10_average": float(np.mean([
                row["target_accuracy"] for row in self.history[-10:]
            ])),
            "runtime_seconds": time.time() - started,
            "run_dir": str(self.run_dir.resolve()),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
