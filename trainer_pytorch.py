import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from datasets import load_dataset
from transformers import AutoTokenizer
from loguru import logger
from dataclasses import dataclass
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from src.model import SmolLM


@dataclass
class MainConfig:
    vocab_size: int = 49152
    emb_dim: int = 576
    intermediate_size: int = 1536
    num_layers: int = 30
    n_q_heads: int = 9
    n_kv_heads: int = 3
    max_seq_len: int = 1024
    dropout: float = 0.1
    rms_norm_eps: float = 1e-05
    init_std: float = 0.041666666666666664


config = MainConfig()


class CustomDataset(Dataset):
    def __init__(self, dataset, config, fraction=0.1):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = config
        self.limit = int(fraction * len(self.dataset))

    def __len__(self):
        # return len(self.dataset)
        return self.limit

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


def setup_logging(log_dir="log"):
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


def train(model, dataloader, config, checkpoint_path="checkpoint", epochs=500):
    # Verify GPU availability
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        logger.error("Insufficient GPUs. Ensure two GPUs are available.")
        return

    # Use DataParallel across both GPUs
    model = DataParallel(model)
    device = torch.device("cuda")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=6e-4)

    # Mixed Precision Setup
    scaler = GradScaler()
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Wrap dataloader with tqdm
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Mixed Precision Training
            with autocast(device_type=device, dtype=torch.float16):
                outputs = model(x.squeeze())
                loss = criterion(outputs.view(-1, outputs.size(-1)), y.flatten())

            # loss.backward()
            # optimizer.step()

            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            total_loss += loss.item()

            # if batch_idx % 100 == 0:
            #     logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # Log prediction periodically
        if epoch != 0 and epoch % 25 == 0:
            log_prediction(model.module, config)

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_file = os.path.join(checkpoint_path, f"smolLM.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                checkpoint_file,
            )
            logger.info(
                f"Saved new best model at epoch {epoch} with loss {best_loss:.4f}"
            )


def log_prediction(model, config):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    prompt = "Which is the fastest"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = greedy_decode(
            model, input_ids, max_length=100, tokenizer=tokenizer
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"Prediction: {generated_text}")


def greedy_decode(model, input_ids, max_length=100, tokenizer=None):
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length - current_ids.shape[1]):
            outputs = model(current_ids)
            last_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return current_ids


if __name__ == "__main__":
    setup_logging()

    # Load dataset
    cosmo_micro_ds = load_dataset("Syed-Hasan-8503/cosmopedia-10k", split="train")
    custom_ds = CustomDataset(cosmo_micro_ds, config, fraction=0.15)
    dataloader = DataLoader(custom_ds, batch_size=8, shuffle=True, num_workers=0)

    # Initialize model
    model = SmolLM(config)

    # Train
    checkpoint_path = "checkpoint"
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = "/kaggle/working/checkpoint"
    train(model, dataloader, config, checkpoint_path)
