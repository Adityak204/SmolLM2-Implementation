## Model Architecture
```
=======================================================================================================================================
Layer (type (var_name))                                      Input Shape               Output Shape              Param #
=======================================================================================================================================
SmolLM (SmolLM)                                              [4, 8192]                 [4, 8192, 49152]          --
├─Embedding (wte)                                            [4, 8192]                 [4, 8192, 576]            28,311,552
├─ModuleList (transformer_blocks)                            --                        --                        --
│    └─TransformerBlock (0)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (1)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (2)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (3)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (4)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (5)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (6)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (7)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (8)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (9)                                  [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (10)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (11)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (12)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (13)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (14)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (15)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (16)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (17)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (18)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (19)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (20)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (21)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (22)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (23)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (24)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (25)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (26)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (27)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (28)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
│    └─TransformerBlock (29)                                 [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (input_layernorm)                        [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─Attention (attention)                            [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (q_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Linear (k_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─Linear (v_proj)                             [4, 8192, 576]            [4, 8192, 192]            110,592
│    │    │    └─LlamaRotaryEmbedding (rotary_embedding)     [4, 9, 8192, 64]          [4, 9, 8192, 64]          --
│    │    │    └─Dropout (attn_dropout)                      [4, 9, 8192, 8192]        [4, 9, 8192, 8192]        --
│    │    │    └─Linear (o_proj)                             [4, 8192, 576]            [4, 8192, 576]            331,776
│    │    │    └─Dropout (resid_dropout)                     [4, 8192, 576]            [4, 8192, 576]            --
│    │    └─RMSNorm (attention_layernorm)                    [4, 8192, 576]            [4, 8192, 576]            576
│    │    └─FeedForward (feed_forward)                       [4, 8192, 576]            [4, 8192, 576]            --
│    │    │    └─Linear (gate_proj)                          [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (up_proj)                            [4, 8192, 576]            [4, 8192, 1536]           884,736
│    │    │    └─Linear (down_proj)                          [4, 8192, 1536]           [4, 8192, 576]            884,736
│    │    │    └─Dropout (dropout)                           [4, 8192, 576]            [4, 8192, 576]            --
├─RMSNorm (layernorm)                                        [4, 8192, 576]            [4, 8192, 576]            576
├─Linear (lm_head)                                           [4, 8192, 576]            [4, 8192, 49152]          28,311,552
=======================================================================================================================================
Total params: 162,826,560
Trainable params: 162,826,560
Non-trainable params: 0
Total mult-adds (M): 651.31
=======================================================================================================================================
Input size (MB): 0.26
Forward/backward pass size (MB): 63015.22
Params size (MB): 651.31
Estimated Total Size (MB): 63666.79
=======================================================================================================================================

```