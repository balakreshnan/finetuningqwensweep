# EOS - H100 - 8GPU Fine tuning Cosmos 3 Nano

## Steps

- list

```
sinfo --summarize
sacctmgr show associations user=$USER format=Account,Partition
squeue -u $USER
```

- Create credential directory

```
mkdir -p ~/.config/enroot
touch ~/.config/enroot/.credentials
```

- create credentials

```
cat > ~/.config/enroot/.credentials << 'EOF'
machine gxxxxxx login xxx@xxxx.com password xxxxxxxxx
machine nvcr.io login $oauthtoken password xxxxxxxx
EOF
```

- Run a interactive container to fine tunie

```
srun --account=general_sa \
     --partition=interactive \
     --nodes=1 \
     --ntasks-per-node=1 \
     --time=2:00:00 \
     --job-name=general_sa-finetune:interactive \
     --container-image=nvcr.io#nvidia/nemo:25.04 \
     --container-mount-home \
     --no-container-remap-root \
     --mpi=pmix \
     --pty bash

```

- install dependencies

```
# Step 2: Install Dependencies (inside container)
pip install --upgrade pip
pip install "numpy<2"
pip install diffusers>=0.32.0 transformers>=4.45.0 accelerate
pip install peft>=0.14.0
pip install wandb pyyaml decord opencv-python-headless huggingface_hub
```

```
pip install transformers==4.51.3 datasets accelerate peft bitsandbytes trl
pip install wandb huggingface_hub
```

- Verify libraries

```
python -c "import diffusers; print(diffusers.__version__)"
python -c "from diffusers import CosmosTransformer3DModel; print('OK')"
python -c "import peft; print(peft.__version__)"
```

- Setup project folders

```
export PROJECT=/lustre/fsw/general_sa/bbalakreshna/cosmos3
mkdir -p $PROJECT/{configs,data,models,checkpoints}
cd $PROJECT
```

```
export HF_TOKEN="xxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxx"
```

- download the dataset
- first create folders to save

```
export BASE_DIR="/lustre/fsw/general_sa/bbalakreshna"

# Dataset
mkdir -p ${BASE_DIR}/cosmos3/data
# Model
mkdir -p ${BASE_DIR}/cosmos3/models
# Checkpoints
mkdir -p ${BASE_DIR}/cosmos3/checkpoints
mkdir -p ${BASE_DIR}/cosmos3/configs
```

- create the download dataset code

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/download_dataset.py << 'EOF'
import os
from huggingface_hub import snapshot_download

# Set your Hugging Face token
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"  # Replace with your actual token

BASE_DIR = "/lustre/fsw/general_sa/bbalakreshna/cosmos3"

# Create directories
os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)

# Download small test dataset first (validate pipeline)
print("📥 Downloading dataset...")
dataset_path = snapshot_download(
    repo_id="sayakpaul/ucf101-subset",
    repo_type="dataset",
    local_dir=f"{BASE_DIR}/data/cosmos3_dataset",
    token=os.environ["HF_TOKEN"]
)
print(f"✅ Dataset downloaded to: {dataset_path}")

# Download Cosmos 3 Nano base model
print("📥 Downloading Cosmos 3 Nano model...")
model_path = snapshot_download(
    repo_id="nvidia/Cosmos-Predict2-2B-Video2World",
    local_dir=f"{BASE_DIR}/models/cosmos3-nano",
    token=os.environ["HF_TOKEN"]
)
print(f"✅ Model downloaded to: {model_path}")

print("\n🎉 All downloads complete!")
print(f"   Dataset: {BASE_DIR}/data/cosmos3_dataset")
print(f"   Model:   {BASE_DIR}/models/cosmos3-nano")
EOF
```

- now execute the code

```
python /lustre/fsw/general_sa/bbalakreshna/cosmos3/download_dataset.py
```

- deflate the dataset

```
cd /lustre/fsw/general_sa/bbalakreshna/cosmos3/data/cosmos3_dataset
tar -xf UCF101_subset.tar.gz
find UCF101_subset -type f -name "*.avi" | wc -l
find UCF101_subset -type d | head -20
```

- Download cosmos model

```
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'nvidia/Cosmos-Predict2-2B',
    local_dir='/lustre/fsw/general_sa/bbalakreshna/cosmos3/models/cosmos3-nano',
)
"

# Verify structure
ls models/cosmos3-nano/
```


- Prepare dataset
- this step is optional onlu run when dataset is not deflated as above.

```
cd $PROJECT/data
mkdir cosmos3_dataset && cd cosmos3_dataset

# Example: UCF101 subset (or use your own .avi/.mp4 videos)
# If downloading from HuggingFace:
pip install huggingface_hub[cli]
huggingface-cli download --repo-type dataset <your-dataset-repo> --local-dir .

# Extract (note: file may be plain tar despite .gz name)
file UCF101_subset.tar.gz
# If "POSIX tar archive" → use tar -xf (NOT tar -xzf)
tar -xf UCF101_subset.tar.gz

# Verify
find UCF101_subset -type f -name "*.avi" | wc -l
# Should show video count (e.g. 407)
```

- create the folder

```
export BASE_DIR="/lustre/fsw/general_sa/bbalakreshna"

# Create the configs directory
mkdir -p ${BASE_DIR}/cosmos3/configs
```

```
export PROJECT="${BASE_DIR}/cosmos3"
echo $PROJECT
```

```
mkdir -p ${PROJECT}/configs
```

- create config yaml file

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/configs/cosmos3_nano_finetune.yaml << 'EOF'
# ═══════════════════════════════════════════════════════════════════════════════
# Cosmos3-Nano LoRA Fine-Tuning Configuration
# ═══════════════════════════════════════════════════════════════════════════════

model:
  # Native Cosmos .pt checkpoint (post-trained base, 3.9GB)
  checkpoint_path: /lustre/fsw/general_sa/bbalakreshna/cosmos3/models/cosmos3-nano/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
  # T5-v1.1-large produces 1024-dim embeddings matching cross_attn kv_dim
  t5_model: google/t5-v1_1-large

peft:
  enabled: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  # Target the attention projections in all 28 transformer blocks
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - output_proj

optimizer:
  lr: 2e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  warmup_steps: 100
  min_lr: 1e-6

trainer:
  train_dir: /lustre/fsw/general_sa/bbalakreshna/cosmos3/data/cosmos3_dataset
  max_epochs: 3
  batch_size: 4
  gradient_clip_val: 1.0
  log_every_n_steps: 10

exp_manager:
  exp_dir: ./checkpoints
  wandb_logger_kwargs:
    project: cosmos3-nano-finetune
    name: cosmos3-nano-lora-h100-8GPU
  hf_hub:
    repo_id: Balab2021/cosmos3-nano-lora
    private: true
    base_model: nvidia/Cosmos-Predict2-2B
EOF
```

- write the finetuning code

```
cat > /lustre/fsw/general_sa/bbalakreshna/cosmos3/finetune_cosmos3_nano.py << 'PYTHON'
"""Cosmos3-Nano LoRA Fine-Tuning — Native Checkpoint + HuggingFace Upload"""
import os
import re
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from huggingface_hub import HfApi, create_repo
    HAS_HF = True
except ImportError:
    HAS_HF = False


# ═══════════════════════════════════════════════════════════════════════════════
# Cosmos3 DiT Architecture (exact shapes from checkpoint)
# 28 blocks, hidden=2048, text_dim=1024, intermediate=8192
# adaln_dim=256, in_channels=72, out_channels=64
# vocab_size=100352 (built-in text embedding), t_out_dim=6144
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._extra_state = None

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight

    def set_extra_state(self, state):
        self._extra_state = state

    def get_extra_state(self):
        return self._extra_state


class Attention(nn.Module):
    def __init__(self, hidden_dim, kv_dim=None, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        kv_dim = kv_dim or hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, context=None):
        ctx = context if context is not None else x
        B, N, C = x.shape
        S = ctx.shape[1]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.output_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.layer2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.layer2(F.silu(self.layer1(x)))


class AdaLNModulation(nn.Module):
    """AdaLN: hidden_dim -> adaln_dim -> num_outputs * hidden_dim"""
    def __init__(self, hidden_dim, adaln_dim, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, adaln_dim, bias=True)
        self.linear2 = nn.Linear(adaln_dim, hidden_dim * num_outputs, bias=True)

    def forward(self, t_emb):
        h = self.linear2(F.silu(self.linear1(t_emb)))
        return h.chunk(self.num_outputs, dim=-1)


class CosmosBlock(nn.Module):
    def __init__(self, hidden_dim, text_dim, intermediate_dim, num_heads, adaln_dim=256):
        super().__init__()
        self.self_attn = Attention(hidden_dim, num_heads=num_heads)
        self.cross_attn = Attention(hidden_dim, kv_dim=text_dim, num_heads=num_heads)
        self.mlp = MLP(hidden_dim, intermediate_dim)
        self.adaln_modulation_self_attn = AdaLNModulation(hidden_dim, adaln_dim, 3)
        self.adaln_modulation_cross_attn = AdaLNModulation(hidden_dim, adaln_dim, 3)
        self.adaln_modulation_mlp = AdaLNModulation(hidden_dim, adaln_dim, 3)

    def forward(self, x, t_emb_sa, t_emb_ca, t_emb_mlp, text_embeds):
        # Self-attention with AdaLN
        scale, shift, gate = self.adaln_modulation_self_attn(t_emb_sa)
        h = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + gate.unsqueeze(1) * self.self_attn(h)

        # Cross-attention with AdaLN
        scale, shift, gate = self.adaln_modulation_cross_attn(t_emb_ca)
        h = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + gate.unsqueeze(1) * self.cross_attn(h, context=text_embeds)

        # MLP with AdaLN
        scale, shift, gate = self.adaln_modulation_mlp(t_emb_mlp)
        h = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + gate.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels, adaln_dim=256):
        super().__init__()
        self.adaln_modulation = AdaLNModulation(hidden_dim, adaln_dim, 2)
        self.linear = nn.Linear(hidden_dim, out_channels, bias=False)

    def forward(self, x, t_emb):
        scale, shift = self.adaln_modulation(t_emb)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    """sinusoidal(2048) -> linear_1(2048->2048) -> SiLU -> linear_2(2048->6144)"""
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, t_emb):
        return self.linear_2(F.silu(self.linear_1(t_emb)))


class CosmosTransformer(nn.Module):
    """
    Cosmos3-Nano DiT
    - 28 blocks, hidden=2048, text_dim=1024, intermediate=8192
    - adaln_dim=256, in_channels=72, out_channels=64
    - Built-in text embedding: vocab=100352 -> 1024
    - t_embedder output=6144 (split 3x2048 for sa/ca/mlp)
    """

    def __init__(self, hidden_dim=2048, text_dim=1024, intermediate_dim=8192,
                 num_blocks=28, num_heads=16, in_channels=72, out_channels=64,
                 adaln_dim=256, vocab_size=100352, t_out_dim=6144):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Patch embedder: Linear(72, 2048)
        self.x_embedder_proj = nn.Linear(in_channels, hidden_dim, bias=False)

        # Built-in text embedding: Linear(100352, 1024)
        self.crossattn_proj = nn.Linear(vocab_size, text_dim, bias=True)

        # Timestep embedder: 2048 -> SiLU -> 6144
        self.t_embedder = TimestepEmbedder(hidden_dim, t_out_dim)

        # Norm applied to sinusoidal embedding
        self.t_embedding_norm = RMSNorm(hidden_dim)

        # Positional embedder buffers
        self.register_buffer('pos_seq', torch.zeros(128))
        self.register_buffer('pos_dim_spatial_range', torch.zeros(21))
        self.register_buffer('pos_dim_temporal_range', torch.zeros(22))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CosmosBlock(hidden_dim, text_dim, intermediate_dim, num_heads, adaln_dim)
            for _ in range(num_blocks)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_dim, out_channels, adaln_dim)

    def get_timestep_embedding(self, timesteps, dim=2048):
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=timesteps.device).float() / half
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, timesteps, text_embeds):
        """
        x:            (B, N, 72)    — patchified latent tokens
        timesteps:    (B,)          — float [0, 1000]
        text_embeds:  (B, S, 1024)  — T5 encoder output
        Returns:      (B, N, 64)    — predicted velocity
        """
        # Embed patches: (B, N, 72) -> (B, N, 2048)
        h = self.x_embedder_proj(x)

        # Timestep: sinusoidal -> norm -> embed -> 6144
        t_sinusoidal = self.get_timestep_embedding(timesteps, self.hidden_dim)
        t_normed = self.t_embedding_norm(t_sinusoidal)
        t_full = self.t_embedder(t_normed)  # (B, 6144)

        # Split into 3 x 2048 for (self_attn, cross_attn, mlp)
        t_emb_sa, t_emb_ca, t_emb_mlp = t_full.chunk(3, dim=-1)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb_sa, t_emb_ca, t_emb_mlp, text_embeds)

        # Final projection: (B, N, 2048) -> (B, N, 64)
        h = self.final_layer(h, t_emb_sa)
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_cosmos_checkpoint(model, ckpt_path):
    print(f"📦 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state_dict = {}
    skipped = []

    for k, v in ckpt.items():
        key = k[4:] if k.startswith('net.') else k

        if not isinstance(v, torch.Tensor):
            skipped.append(key)
            continue

        # Map checkpoint keys to our module keys
        key = key.replace('x_embedder.proj.1.', 'x_embedder_proj.')
        key = key.replace('crossattn_proj.0.', 'crossattn_proj.')
        key = key.replace('t_embedder.1.', 't_embedder.')
        key = key.replace('pos_embedder.seq', 'pos_seq')
        key = key.replace('pos_embedder.dim_spatial_range', 'pos_dim_spatial_range')
        key = key.replace('pos_embedder.dim_temporal_range', 'pos_dim_temporal_range')

        # AdaLN: .1. -> .linear1.  and  .2. -> .linear2.
        key = re.sub(r'adaln_modulation_(self_attn|cross_attn|mlp)\.1\.', r'adaln_modulation_\1.linear1.', key)
        key = re.sub(r'adaln_modulation_(self_attn|cross_attn|mlp)\.2\.', r'adaln_modulation_\1.linear2.', key)
        key = re.sub(r'final_layer\.adaln_modulation\.1\.', 'final_layer.adaln_modulation.linear1.', key)
        key = re.sub(r'final_layer\.adaln_modulation\.2\.', 'final_layer.adaln_modulation.linear2.', key)

        state_dict[key] = v

    if skipped:
        print(f"   Skipped {len(skipped)} non-tensor entries: {skipped}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   ⚠️  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    loaded = len(state_dict) - len(unexpected)
    print(f"   ✅ Loaded {loaded}/{len(state_dict)} tensors")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class CosmosVideoDataset(Dataset):
    VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}

    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = sorted([
            f for f in self.data_dir.rglob("*")
            if f.suffix.lower() in self.VIDEO_EXTS
        ])
        print(f"📂 Dataset: {len(self.samples)} videos from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def _filename_to_prompt(self, path):
        name = path.stem
        name = re.sub(r"^v_", "", name)
        name = re.sub(r"_g\d+_c\d+$", "", name)
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()
        return f"a video of {words}"

    def __getitem__(self, idx):
        prompt = self._filename_to_prompt(self.samples[idx])
        tokens = self.tokenizer(
            prompt, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "prompt": prompt,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build & load Cosmos transformer ──
    model_cfg = config["model"]
    ckpt_path = model_cfg["checkpoint_path"]

    transformer = CosmosTransformer(
        hidden_dim=2048, text_dim=1024, intermediate_dim=8192,
        num_blocks=28, num_heads=16, in_channels=72, out_channels=64,
        adaln_dim=256, vocab_size=100352, t_out_dim=6144,
    )
    transformer = load_cosmos_checkpoint(transformer, ckpt_path)
    transformer = transformer.to(dtype=torch.bfloat16, device=device)

    # ── Apply LoRA ──
    peft_cfg = config["peft"]
    lora_config = LoraConfig(
        r=peft_cfg["lora_rank"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # ── Load T5 text encoder (1024-dim output) ──
    t5_path = model_cfg.get("t5_model", "google/t5-v1_1-large")
    print(f"📦 Loading T5 encoder: {t5_path}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # ── Dataset & DataLoader ──
    train_dir = config["trainer"]["train_dir"]
    batch_size = config["trainer"]["batch_size"]
    dataset = CosmosVideoDataset(train_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── Optimizer ──
    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=float(opt_cfg.get("lr", 2e-5)),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    max_epochs = config["trainer"]["max_epochs"]
    total_steps = len(dataloader) * max_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1),
        eta_min=float(config.get("scheduler", {}).get("min_lr", 1e-6)),
    )

    # Latent dimensions
    in_channels = 72
    out_channels = 64
    seq_len = 1024  # flattened spatial-temporal patches

    print(f"\n🚀 Training: {max_epochs} epochs, {len(dataloader)} steps/epoch")
    print(f"   Trainable: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")
    print(f"   Total:     {sum(p.numel() for p in transformer.parameters()):,}")

    global_step = 0
    last_loss = 0.0

    for epoch in range(max_epochs):
        transformer.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bs = input_ids.shape[0]

            # Encode text -> (B, S, 1024)
            with torch.no_grad():
                text_embeds = text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state

            # Random latent tokens (simulating patchified VAE output)
            latents = torch.randn(bs, seq_len, in_channels,
                                  device=device, dtype=torch.bfloat16)

            # Flow matching: sample timesteps
            timesteps = torch.rand(bs, device=device) * 1000.0
            noise = torch.randn(bs, seq_len, out_channels,
                                device=device, dtype=torch.bfloat16)

            # Target: noise in output space (velocity prediction)
            target = noise

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = transformer(latents, timesteps, text_embeds)

            # Loss
            loss = F.mse_loss(pred.float(), target.float())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                transformer.parameters(),
                config["trainer"].get("gradient_clip_val", 1.0),
            )
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            last_loss = loss.item()
            epoch_loss += last_loss

            # Logging
            if global_step % config["trainer"].get("log_every_n_steps", 10) == 0:
                lr = lr_scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch+1}/{max_epochs}] Step {global_step} | "
                      f"Loss: {last_loss:.4f} | LR: {lr:.2e}")
                if HAS_WANDB:
                    wandb.log({"loss": last_loss, "lr": lr, "epoch": epoch + 1,
                               "step": global_step})

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"  ── Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save LoRA weights locally
    # ═══════════════════════════════════════════════════════════════════════
    save_dir = os.path.join(config["exp_manager"]["exp_dir"], "lora_weights")
    os.makedirs(save_dir, exist_ok=True)
    transformer.save_pretrained(save_dir)
    print(f"\n✅ Training complete! LoRA weights saved to: {save_dir}")

    # ═══════════════════════════════════════════════════════════════════════
    # Upload to HuggingFace Hub
    # ═══════════════════════════════════════════════════════════════════════
    hf_cfg = config.get("exp_manager", {}).get("hf_hub", {})
    hf_repo = hf_cfg.get("repo_id", None)

    if hf_repo and HAS_HF:
        print(f"\n📤 Uploading LoRA weights to HuggingFace Hub: {hf_repo}")

        # Fix PEFT auto-generated README: replace local path with HF model id
        hf_base_model = hf_cfg.get("base_model", "nvidia/Cosmos-Predict2-2B")
        readme_path = os.path.join(save_dir, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                readme = f.read()
            readme = re.sub(r"base_model:.*", f"base_model: {hf_base_model}", readme)
            with open(readme_path, "w") as f:
                f.write(readme)
            print(f"   Fixed README.md base_model -> {hf_base_model}")

        api = HfApi()
        create_repo(hf_repo, exist_ok=True, private=hf_cfg.get("private", True))
        api.upload_folder(
            folder_path=save_dir,
            repo_id=hf_repo,
            commit_message=f"LoRA weights - {max_epochs} epochs, final loss {last_loss:.4f}",
        )
        print(f"✅ Uploaded to https://huggingface.co/{hf_repo}")
    elif hf_repo and not HAS_HF:
        print("⚠️  huggingface_hub not installed — skipping upload. "
              "Run: pip install huggingface_hub")
    else:
        print("ℹ️  No hf_hub.repo_id in config — skipping HuggingFace upload")

    if HAS_WANDB:
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "configs",
                               "cosmos3_nano_finetune.yaml")
    print(f"📋 Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if HAS_WANDB:
        wb_cfg = config.get("exp_manager", {}).get("wandb_logger_kwargs", {})
        wandb.init(
            project=wb_cfg.get("project", "cosmos3-finetune"),
            name=wb_cfg.get("name", "run"),
        )
        print(f"✅ wandb initialized: {wandb.run.url}")

    train(config)
PYTHON
```

- login into wandb

```
wandb login
```

- now run the code

```
cd $PROJECT
```

- changed directory

```
cd /lustre/fsw/general_sa/bbalakreshna/cosmos3
```

```
rm -rf /usr/local/lib/python3.12/dist-packages/transformers*
pip install transformers==4.51.3
```

- validate

```
python -c "from transformers import HybridCache; print('✅ HybridCache OK')"
python -c "from diffusers import CosmosTransformer3DModel; print('✅ Cosmos OK')"
```

```
python finetune_cosmos3_nano.py
```

```
https://wandb.ai/balabala76/cosmos3-nano-finetune/runs/bpxl1y8v
```

