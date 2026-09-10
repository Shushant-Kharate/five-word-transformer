"""Section 4 tests for loss, optimizer updates, logging, and checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset import get_datasets, get_or_build_tokenizer
from train import (
    get_loss_function,
    get_model,
    get_optimizer,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_batch,
    train_one_epoch,
)


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_config()
        cls.device = torch.device("cpu")
        cls.tokenizer = get_or_build_tokenizer(cls.config)
        train_dataset, _, _ = get_datasets(cls.config, cls.tokenizer)
        cls.small_loader = DataLoader(
            Subset(train_dataset, range(4)), batch_size=4, shuffle=False
        )
        cls.batch = next(iter(cls.small_loader))
        cls.vocabulary_size = cls.tokenizer.get_vocab_size()

    def make_components(self):
        set_seed(42)
        model = get_model(self.config, self.vocabulary_size).to(self.device)
        optimizer = get_optimizer(self.config, model)
        loss_function = get_loss_function(
            self.config, self.tokenizer, self.device
        )
        return model, optimizer, loss_function

    def test_one_batch_loss_is_finite_and_updates_weights(self) -> None:
        model, optimizer, loss_function = self.make_components()
        before = model.projection_layer.proj.weight.detach().clone()
        loss = train_one_batch(
            model,
            self.batch,
            optimizer,
            loss_function,
            self.vocabulary_size,
            self.device,
        )
        after = model.projection_layer.proj.weight.detach()
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.equal(before, after))

    def test_pad_position_is_ignored_by_loss(self) -> None:
        _, _, loss_function = self.make_components()
        labels = self.batch["label"]
        logits = torch.randn(4, 7, self.vocabulary_size)
        original_loss = loss_function(
            logits.view(-1, self.vocabulary_size), labels.view(-1)
        )
        modified_logits = logits.clone()
        modified_logits[:, -1, :] = 10_000.0
        modified_loss = loss_function(
            modified_logits.view(-1, self.vocabulary_size), labels.view(-1)
        )
        self.assertTrue(torch.allclose(original_loss, modified_loss))

    def test_one_batch_epoch_writes_tensorboard_log(self) -> None:
        model, optimizer, loss_function = self.make_components()
        with tempfile.TemporaryDirectory() as directory:
            writer = SummaryWriter(directory)
            average_loss, global_step = train_one_epoch(
                model,
                self.small_loader,
                optimizer,
                loss_function,
                self.vocabulary_size,
                self.device,
                epoch=0,
                global_step=0,
                writer=writer,
                show_progress=False,
            )
            writer.close()
            event_files = list(Path(directory).glob("events.out.tfevents.*"))

        self.assertGreater(average_loss, 0.0)
        self.assertEqual(global_step, 1)
        self.assertEqual(len(event_files), 1)

    def test_checkpoint_round_trip(self) -> None:
        model, optimizer, loss_function = self.make_components()
        loss = train_one_batch(
            model,
            self.batch,
            optimizer,
            loss_function,
            self.vocabulary_size,
            self.device,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=3,
                global_step=112,
                training_loss=loss,
            )

            restored_model, restored_optimizer, _ = self.make_components()
            next_epoch, global_step, restored_loss = load_checkpoint(
                checkpoint_path,
                restored_model,
                restored_optimizer,
                self.device,
            )

        self.assertEqual(next_epoch, 4)
        self.assertEqual(global_step, 112)
        self.assertAlmostEqual(restored_loss, loss)
        for original, restored in zip(
            model.parameters(), restored_model.parameters()
        ):
            self.assertTrue(torch.equal(original, restored))


if __name__ == "__main__":
    unittest.main()
