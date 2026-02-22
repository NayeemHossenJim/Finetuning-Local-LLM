<div align="center">

# 🦙 Finetuning Local LLMs with Unsloth

**Efficient LoRA finetuning of large language models — locally, fast, and memory-friendly.**

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="https://unsloth.ai/"><img src="https://img.shields.io/badge/Unsloth-Faster_Finetuning-FFAA00?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAADASURBVHgBnVLBCYQwEBxDLMAWYgnWYCnBFqwgWIIt2IKxBEuwBluIIbhv3oUkqHd5kIXNzmxmsiHEGCMiopSyMefcReScM621iMhaXkRERNu2iYh475mZiYh477GIOOdijDHmnDMRERFRSokxRkRERFQ65xydc8YYIyLqvffWWiMiIqJSSg0hhBBijBFj/R0opVRrrYiIiEopNYQQQogxRoz1dwAAABJRU5ErkJggg==" alt="Unsloth"/></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/badge/🤗_Hugging_Face-Models-FFD21E?style=for-the-badge" alt="Hugging Face"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License: MIT"/></a>

<br/><br/>

<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png" width="300" alt="Unsloth Logo"/>

</div>

---

## Overview

This repository contains two Jupyter notebooks demonstrating how to finetune local LLMs using [Unsloth](https://unsloth.ai/) — a library that makes finetuning significantly faster and less memory-intensive compared to standard HuggingFace training.

Both notebooks follow the same general workflow:

1. Load a pretrained base model with 4-bit quantization
2. Attach LoRA adapters to the model
3. Prepare a task-specific dataset
4. Train with SFT (Supervised Fine-Tuning)
5. Run inference
6. Save the LoRA adapter and/or export to GGUF for local inference with Ollama / llama.cpp

---

## Notebooks

### 1. `ascii_art_completion.ipynb` — Completion Finetuning

> Finetunes a base model to generate ASCII art from scratch (completion-style, no prompt).

| Property | Details |
|---|---|
| **Base Model** | [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| **Unsloth Model** | [`unsloth/Llama-3.2-3B`](https://huggingface.co/unsloth/Llama-3.2-3B) |
| **Dataset** | [`pookie3000/ascii-cats`](https://huggingface.co/datasets/pookie3000/ascii-cats) |
| **Task** | Unconditional ASCII art completion |
| **Quantization** | Full precision (no 4-bit) |
| **Output (LoRA)** | [`Jim1892/Llama-3.2-3B-ascii-cats-bd`](https://huggingface.co/Jim1892/Llama-3.2-3B-ascii-cats-bd) |
| **Output (GGUF)** | [`Jim1892/Llama-3.2-3B-ascii-cats-bd-F32-GGUF`](https://huggingface.co/Jim1892/Llama-3.2-3B-ascii-cats-bd-F32-GGUF) |

---

### 2. `conversation_finetune.ipynb` — Instruction / Chat Finetuning

> Finetunes an instruction-tuned model on a custom conversation dataset to adapt its chat behavior.

| Property | Details |
|---|---|
| **Base Model** | [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| **Unsloth Model** | [`unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit) |
| **Dataset** | [`pookie3000/pg_chat`](https://huggingface.co/datasets/pookie3000/pg_chat) |
| **Task** | Multi-turn conversation / instruction following |
| **Quantization** | 4-bit (bitsandbytes) during training |
| **Output (LoRA)** | [`Jim1892/Meta-Llama-3.1-8B-Instruct-Jim-BD`](https://huggingface.co/Jim1892/Meta-Llama-3.1-8B-Instruct-Jim-BD) |
| **Output (GGUF)** | `Meta-Llama-3.1-8B-q4_k_m-Jim--guide-GGUF` *(q4_k_m quantization)* |

---

## How It Works

### LoRA (Low-Rank Adaptation)

Instead of updating all model weights, LoRA injects small trainable matrices into specific layers. This makes finetuning:

- **Memory efficient** — only a fraction of parameters are trained
- **Fast** — fewer gradients to compute
- **Portable** — the LoRA adapter is a small file that sits on top of the base model

LoRA is applied to the attention and MLP projection layers:

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Key hyperparameters used:

| Parameter | Value | Description |
|---|---|---|
| `r` | 16 | Rank of the LoRA matrices |
| `lora_alpha` | 16 | Scaling factor for LoRA weights |
| `lora_dropout` | 0 | Dropout on LoRA layers |
| `use_rslora` | False | Whether to use rank-stabilized LoRA |
| `gradient_checkpointing` | `"unsloth"` | Reduces VRAM for long contexts |

### Training Configuration

Both notebooks share the same SFT training setup:

| Parameter | Value |
|---|---|
| Batch size | 2 per device |
| Gradient accumulation | 4 steps |
| Max steps | 60 |
| Learning rate | 2e-4 |
| Optimizer | `adamw_8bit` |
| LR scheduler | Linear |
| Max sequence length | 2048 |

### Saving & Export Options

After training, the notebooks demonstrate two saving strategies:

| Format | Use Case | Tool |
|---|---|---|
| **LoRA Adapter** (safetensors) | Resume finetuning or inference via HuggingFace | `model.push_to_hub()` |
| **GGUF** | Local inference with [Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggerganov/llama.cpp) | `model.push_to_hub_gguf()` |

**Quantization options for GGUF export:**

| Method | Bits | Memory | Notes |
|---|---|---|---|
| `q4_k_m` | 4-bit | Low | Default for Ollama pulls |
| `q8_0` | 8-bit | Medium | Higher quality |
| `f16` | 16-bit | High | Near lossless |
| `F32` | 32-bit | Very high | Full precision |

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- A [HuggingFace account](https://huggingface.co/) with access to Llama models

### Installation

The notebooks handle package installation automatically in the first cell. The core dependencies are:

```bash
pip install --no-deps bitsandbytes accelerate xformers peft trl triton
pip install --no-deps cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install --no-deps unsloth
```

### Environment Variables

Create a `.env` file in the project root:

```env
HF_ACCESS_TOKEN=hf_your_read_token_here
HF_ACCESS_TOKEN_WRITE=hf_your_write_token_here
```

- `HF_ACCESS_TOKEN` — Read token for downloading gated models (e.g. Llama)
- `HF_ACCESS_TOKEN_WRITE` — Write token for pushing finetuned models to the Hub

> You can generate tokens at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Running the Notebooks

Open either notebook in VS Code or JupyterLab and run the cells top to bottom:

```
ascii_art_completion.ipynb    ← Completion finetuning on ASCII art
conversation_finetune.ipynb   ← Chat finetuning on conversation data
```

---

## Project Structure

```
Finetuning-Local-LLM/
├── ascii_art_completion.ipynb    # Notebook 1: ASCII art completion finetuning
├── conversation_finetune.ipynb   # Notebook 2: Conversation / instruction finetuning
├── .env                          # API tokens (not committed)
├── LICENSE
└── README.md
```

---

## References & Resources

- [Unsloth Documentation](https://unsloth.ai/) — Core finetuning library used throughout
- [Unsloth GitHub](https://github.com/unslothai/unsloth) — Source and additional notebooks
- [PEFT / LoRA Docs](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/lora#peft.LoraConfig) — LoRA configuration reference
- [TRL / SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — Supervised finetuning trainer
- [HuggingFace Text Generation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/text_generation) — Generation config docs
- [Ollama](https://ollama.com/) — Run GGUF models locally
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inference with GGUF format

---

<div align="center">

*Adapted from [Unsloth notebooks](https://unsloth.ai/). If something breaks, check the upstream notebooks first.*

</div>
