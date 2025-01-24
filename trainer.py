from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import datetime
from src.model import SmolLM2
from dataclasses import dataclass
from loguru import logger
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from loguru import logger


@dataclass
class MainConfig:
    vocab_size: int = 49152
    emb_dim: int = 576
    intermediate_size: int = 1536
    num_layers: int = 10
    n_q_heads: int = 9
    n_kv_heads: int = 3
    max_seq_len: int = 8192
    dropout: float = 0.1
    rms_norm_eps: float = 1e-05
    init_std: float = 0.041666666666666664


config = MainConfig()

cosmo_micro_ds = load_dataset("Syed-Hasan-8503/cosmopedia-10k", split="train")


## Custom Dataset on top of Huggingface's Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config.max_seq_len,
        )
        x = inputs["input_ids"][:, :-1]
        y = inputs["input_ids"][:, 1:]
        return x, y


## DataLoader
custom_ds = CustomDataset(cosmo_micro_ds, config)
dataloader = DataLoader(custom_ds, batch_size=4, shuffle=True)


## Training using Pytorch Lightning
class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        logger.info(f"\n{'='*20} Epoch {trainer.current_epoch} {'='*20}")


class SmolLit(LightningModule):
    def __init__(self, model, config, checkpoint_path="kaggle/working"):
        super().__init__()
        self.model = model
        self.config = config
        self.best_loss = np.inf
        self.checkpoint_path = checkpoint_path

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), y.flatten()
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["train_loss"].mean()
        logger.info(f"Epoch {self.current_epoch} | Avg Loss: {avg_loss:.4f}")

        # Save model checkpoint if loss decreases
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.trainer.save_checkpoint(f"{self.checkpoint_path}/smolLM.ckpt")
            logger.info(f"New best loss: {avg_loss:.4f}, saving model checkpoint")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=5e-4)

    def train_dataloader(self):
        return dataloader


def setup_logging(log_dir="kaggle/working"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
        colorize=True,
        level="INFO",
    )

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="30 days",
    )

    logger.info(f"Logging setup complete. Logs will be saved to: {log_file}")


def main(model):
    setup_logging()
    smol_lit = SmolLit(model, config)
    checkpoint_path = "kaggle/working"

    latest_checkpoint = None
    progress_bar = CustomProgressBar()
    trainer = Trainer(
        max_epochs=5000,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision=16,
        callbacks=[progress_bar],
        enable_progress_bar=True,
    )

    if latest_checkpoint:
        logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.fit(model, checkpoint_path=latest_checkpoint)
    else:
        logger.info("Starting training from scratch")
        trainer.fit(model)


if __name__ == "__main__":
    model = SmolLM2(config)
    main(model)
