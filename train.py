import glob
import math
import re
import time
from typing import Tuple
import warnings

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TokenDataset
from model import GLaMA, ModelConfig
import torch
from torch.amp import autocast, GradScaler
import tiktoken

from torchmetrics.text import Perplexity
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

torch.manual_seed(24)
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = tiktoken.get_encoding("gpt2")

train_file_path = "train.bin"
val_file_path = "val.bin"
gradient_accumulation_steps = 4
lr = 6e-4 # max learning rate
max_iters = 15258 # number of batches // grad accumulation steps
lr_decay_iters = 15258
warmup_iters = 400 # ~2.5% of max_iters
min_lr = 6e-5 # min learning rate
grad_clip = 1.0 # clip gradient at this value
eval_interval = 500
eval_iter = 100

def get_model_path() -> str | None:
    # Get latest model path
    model_path = glob.glob("models/glama_*.pth")
    if len(model_path) == 0:
        return None
    
    return max(model_path, key=lambda x: int(re.search(r'(\d+)', x).group()))

def create_doc_ids(tokens: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    tokens: (batch, seq_len)
    Returns: (batch, seq_len) with document IDs
    """
    # Find EOS positions
    is_eos = (tokens == eos_token_id)
    # Cumsum to create incrementing doc IDs
    # Subtract is_eos so the separator itself is grouped with the document it ENDS
    doc_ids = is_eos.cumsum(dim=1) - is_eos.long()
    return doc_ids

@torch.no_grad()
def evaluate_model(model: GLaMA, train_loader: DataLoader, val_loader: DataLoader, global_steps: int) -> Tuple[float, float, float]:
    model.eval()

    train_losses = torch.zeros(eval_iter)
    for i, (x, y) in enumerate(train_loader):
        if i >= eval_iter:
            break

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        doc_ids = create_doc_ids(x, config.eos_token_id)

        with autocast(device_type=device, dtype=config.dtype):
            _, loss = model(x, y, doc_ids)
            
        train_losses[i] = loss.item()        

    val_losses = torch.zeros(min(eval_iter, len(val_loader)))
    perp = Perplexity(ignore_index=-100).to(device)

    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iter:
            break

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        doc_ids = create_doc_ids(x, config.eos_token_id)

        with autocast(device_type=device, dtype=config.dtype):
            logits, loss = model(x, y, doc_ids)
        
        perp.update(logits, y)
        val_losses[i] = loss.item()        

    pplx = perp.compute()

    # generate one sentence
    generated = model.generate(torch.tensor(tokenizer.encode("Hello! I am "), device=device).unsqueeze(0), max_new_tokens=256, temperature=0.6, top_k=20)
    with open("generated.txt", "a") as f:
        f.write(f"Steps: {global_steps}\n")
        f.write(tokenizer.decode(generated[0].tolist()) + "\n")
        f.write("========================================\n\n")

    model.train()
    return train_losses.mean().item(), val_losses.mean().item(), pplx.item()

# Model config
config = ModelConfig()

print(f"dtype: {config.dtype}")

# Initialize model
model = GLaMA(config)

# move to device
model.to(device)

# create optimizer
optimizer = model.init_optimizer(weight_decay=config.weight_decay, lr=lr, device=device)

# init scheduler
scheduler = model.init_scheduler(optimizer, warmup_iters, lr_decay_iters, min_lr, lr)

global_steps = 0
tokens_seen = 0
best_val_loss = float('inf')

# load from file if available
model_path = get_model_path()
try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    # initial_epoch = int(checkpoint["epoch"])
    global_steps = int(checkpoint["global_steps"])
    tokens_seen = int(checkpoint["tokens_seen"])
    best_val_loss = float(checkpoint["best_val_loss"])
    print(f"Loaded model from {model_path}")
except (FileNotFoundError, Exception) as e:
    print(f"Initializing new model")

# compile model
raw_model = model
model = torch.compile(model)

print("Number of parameters: %.2fM" % (raw_model.get_num_params/1e6,))

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = GradScaler(enabled=(config.dtype == torch.float16))

train_dataset = TokenDataset(train_file_path, config.seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
)

val_dataset = TokenDataset(val_file_path, config.seq_len)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True
)

# should have a separate eval dataset
eval_train_ds = TokenDataset(train_file_path, config.seq_len)
eval_train_loader = DataLoader(
    eval_train_ds,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True
)

torch.cuda.empty_cache()

batch_iter = tqdm(train_loader, desc="Processing...")
accum_loss = 0.0
micro_steps = 0
t0 = time.time()
start_time = time.time()
running_mfu = -1.0

for batch in batch_iter:
    x, y = batch
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    doc_ids = create_doc_ids(x, config.eos_token_id)

    with autocast(device_type=device, dtype=config.dtype):
        logits, loss = model(x, y, doc_ids)
        accum_loss += loss.item()
        loss = loss / gradient_accumulation_steps

    # backpropagate loss
    scaler.scale(loss).backward()

    micro_steps += 1

    # gradient accumulation
    if micro_steps % gradient_accumulation_steps == 0:

        # gradient clipping
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # update weights
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        mfu = raw_model.estimate_mfu(config.batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        # calcualte tokens per second
        tokens_per_second = config.batch_size * gradient_accumulation_steps * config.seq_len // dt

        batch_iter.set_postfix({"loss": accum_loss / gradient_accumulation_steps, "mfu": f"{running_mfu * 100:.2f}", "tps": f"{tokens_per_second:.2f}"})
        accum_loss = 0.0

        # log learning rate
        writer.add_scalar("learning rate", scheduler.get_last_lr()[0], global_steps)
        # writer.flush()

        global_steps += 1
        micro_steps = 0        
        
        # evaluate model
        if global_steps % eval_interval == 0 and global_steps > 0:
            train_loss, val_loss, pplx = evaluate_model(model, eval_train_loader, val_loader, global_steps)
            
            # log metrics
            writer.add_scalar("train loss", train_loss, global_steps)
            writer.add_scalar("val loss", val_loss, global_steps)
            writer.add_scalar("perplexity", pplx, global_steps)
            writer.flush()

            if val_loss < best_val_loss and global_steps > 1000:
                # save model
                torch.save({
                    'model_state': raw_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'global_steps': global_steps,
                    'tokens_seen': tokens_seen,
                    'best_val_loss': val_loss
                }, f"models/glama_{global_steps}.pth")

                best_val_loss = val_loss

    tokens_seen += x.numel()

writer.flush()
writer.close()

print(f"Training finished in : {(time.time() - start_time) / 60:.2f} minutes")

# save model
print("Saving the model...")

torch.save({
    'model_state': raw_model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'global_steps': global_steps,
    'tokens_seen': tokens_seen,
    'best_val_loss': best_val_loss
}, f"models/glama_{global_steps}.pth")

print(f"Model saved successfully in: models/glama_{global_steps}.pth")
