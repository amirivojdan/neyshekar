"""Mean-loss ASR models must not multiply gradients when accumulation changes."""

import tempfile
import unittest

import torch
from transformers import Trainer, TrainingArguments

from neyshekar_experiments.training import configure_mean_loss_model


class MeanLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_values, labels, **kwargs):
        return {"loss": ((input_values * self.weight - labels) ** 2).mean()}


class GradientAccumulationTests(unittest.TestCase):
    def train_once(self, batch_size):
        model = configure_mean_loss_model(MeanLossModel())
        examples = [
            {"input_values": torch.tensor([float(value)]), "labels": torch.tensor([2.0 * value])}
            for value in (1, 2, 3, 4)
        ]
        with tempfile.TemporaryDirectory() as output:
            arguments = TrainingArguments(
                output_dir=output,
                max_steps=1,
                use_cpu=True,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4 // batch_size,
                max_grad_norm=0.0,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=examples,
                optimizers=(torch.optim.SGD(model.parameters(), lr=0.1), None),
            )
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            trainer.train()
        return float(model.weight.detach())

    def test_one_full_batch_equals_accumulated_microbatches(self):
        full = self.train_once(4)
        accumulated = self.train_once(2)
        self.assertAlmostEqual(full, 3.0, places=5)
        self.assertAlmostEqual(accumulated, full, places=5)
