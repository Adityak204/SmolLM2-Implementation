import os
from datetime import datetime
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from loguru import logger
from dataclasses import dataclass

from src.model import SmolLM2


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


cosmo_micro_ds = load_dataset("Syed-Hasan-8503/cosmopedia-10k", split="train")
custom_ds = CustomDataset(cosmo_micro_ds, config)
dataloader = DataLoader(custom_ds, batch_size=4, shuffle=True)


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        logger.info(f"\n{'='*20} Epoch {trainer.current_epoch} {'='*20}")


class SmolLit(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.best_loss = np.inf
        self.checkpoint_path = "kaggle/working"
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), y.flatten()
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log prediction every 500 epochs
        if self.current_epoch % 500 == 0:
            self.log_prediction()

        return loss

    def log_prediction(self):
        # Greedy decoding for the prompt
        prompt = "Which is the fastest"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate using greedy decoding
        generated_ids = self.greedy_decode(input_ids)

        # Decode and log the prediction
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        logger.info(f"Epoch {self.current_epoch} Prediction: {generated_text}")

    def greedy_decode(self, input_ids, max_length=100):
        self.model.eval()
        current_ids = input_ids

        with torch.no_grad():
            for _ in range(max_length - current_ids.shape[1]):
                # Get model outputs
                outputs = self.model(current_ids)

                # Get the last token prediction
                last_token_logits = outputs[:, -1, :]

                # Get the most likely next token
                next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)

                # Append the most likely token
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Stop if end of sequence token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return current_ids

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
    model = SmolLM2(config)
    smol_lit = SmolLit(model, config)

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

    logger.info("Starting training from scratch")
    trainer.fit(model)


if __name__ == "__main__":
    model = SmolLM2(config)
    main(model)
