import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
from dataclasses import dataclass

from src.model import SmolLM


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


def generate_prediction(model, prompt, max_length=100):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = greedy_decode(
            model, input_ids, max_length=max_length, tokenizer=tokenizer
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    # Set page configuration
    st.set_page_config(page_title="SmolLM2-TextGen", page_icon="🤖")

    # Title and description
    st.title("SmolLM2-TextGen 🤖")
    st.write("Generate text using the SmolLM2 language model")

    # Load the model (you'll need to replace this with your actual model loading logic)
    @st.cache_resource
    def load_model(config):
        model = SmolLM(config)
        return model

    # Try to load the model
    try:

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
        model = load_model(config)
        # load checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = "/Users/aditya/Documents/self_learning/ERA V3/week 13/artifacts/m1/smolLM-v2.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)[
            "model_state_dict"
        ]
        model.load_state_dict(checkpoint)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Input prompt
    prompt = st.text_input(
        "Enter your prompt:", placeholder="Type a sentence to generate text..."
    )

    # Max length slider
    max_length = st.slider(
        "Maximum Generation Length", min_value=10, max_value=200, value=100, step=10
    )

    # Generate button
    if st.button("Generate Text"):
        if not prompt:
            st.warning("Please enter a prompt.")
            return

        # Show loading spinner
        with st.spinner("Generating text..."):
            try:
                # Generate text
                generated_text = generate_prediction(model, prompt, max_length)

                # Display generated text
                st.subheader("Generated Text:")
                st.write(generated_text)

            except Exception as e:
                st.error(f"An error occurred during text generation: {e}")


if __name__ == "__main__":
    main()
