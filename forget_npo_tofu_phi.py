import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
import argparse
import random
import matplotlib.pyplot as plt

def plot_loss(loss_history, save_path):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("NPO loss")
    plt.title("TOFU NPO Training Loss")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

from tofu_data_module import TOFUIndexedDataset, tofu_custom_collator_forget

def npo_loss(policy_logps, ref_logps, beta=0.05):
    log_ratio = (ref_logps - policy_logps).clamp(-5, 5)
    loss = F.softplus(-beta * log_ratio)
    return loss.mean()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_batch_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    mask = (shift_labels != -100)
    token_counts = mask.sum(dim=-1).clamp(min=1)
    log_probs = -(loss * mask).sum(dim=-1) / token_counts
    return log_probs

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/phi-1_5")
    parser.add_argument("--data_dir", type=str, default="./tofu_data")
    parser.add_argument("--forget_pct", type=int, default=10)
    parser.add_argument("--coreset_pct", type=float, default=10.0)
    parser.add_argument("--coreset_method", type=str, default="random", choices=["random", "embedding_kcenter"],
                        help="Coreset selection method used when generating index file")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.05, help="NPO temperature")
    parser.add_argument("--retain_weight", type=float, default=0.1, help="Weight for retain KL loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./models/phi_npo_tofu")
    parser.add_argument("--mixed_precision", action="store_true", help="启用混合精度训练")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    parser.add_argument("--grad_acc_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit"],
                        help="优化器类型 (adamw 或 adamw8bit，需安装 bitsandbytes)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点以减少显存")
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 tokenizer
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    # 可选：梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    print(f"Model parameter dtype: {next(model.parameters()).dtype}")

    # 2. 参考模型（保持在CPU）
    print("Reference model stays on CPU...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).cpu()
    ref_model.eval()

    # 3. 数据集
    # 构建索引文件名（包含方法名）
    index_file = os.path.join(
        args.data_dir,
        "indices",
        f"tofu_forget{args.forget_pct:02d}_{args.coreset_method}_{args.coreset_pct}_{args.seed}.json"
    )
    if not os.path.exists(index_file):
        # 向后兼容：尝试随机方法的老文件名
        old_index_file = os.path.join(
            args.data_dir,
            "indices",
            f"tofu_forget{args.forget_pct:02d}_random_{args.coreset_pct}_{args.seed}.json"
        )
        if os.path.exists(old_index_file):
            index_file = old_index_file
            print(f"Warning: Preferred index file not found, using old random index: {old_index_file}")
        else:
            print(f"Warning: Index file not found at {index_file}, using full forget set.")
            index_file = None

    dataset = TOFUIndexedDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        forget_percentage=args.forget_pct,
        index_file=index_file,
        mode="train"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tofu_custom_collator_forget
    )
    print(f"Dataset size (forget): {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    # 4. 优化器（支持8-bit）
    if args.optimizer == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(model.parameters(), lr=args.lr)
            print("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            print("bitsandbytes not installed, falling back to standard AdamW")
            optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = len(dataloader) * args.epochs // args.grad_acc_steps  # 注意累积后更新次数减少
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    # 5. 训练循环
    print("Starting NPO Training...")
    model.train()
    loss_history = []

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            (f_input_ids, f_labels, f_mask), \
            (r_input_ids, r_labels, r_mask), _ = batch

            f_input_ids = f_input_ids.to(device)
            f_labels    = f_labels.to(device)
            f_mask      = f_mask.to(device)

            r_input_ids = r_input_ids.to(device)
            r_labels    = r_labels.to(device)
            r_mask      = r_mask.to(device)

            # ---------------------------
            # 遗忘部分：NPO 损失
            # ---------------------------
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                policy_outputs = model(f_input_ids, attention_mask=f_mask, use_cache=False)
                policy_logps = get_batch_logps(policy_outputs.logits, f_labels)

            with torch.no_grad():
                ref_outputs = ref_model(f_input_ids.cpu(), attention_mask=f_mask.cpu(), use_cache=False)
                ref_logps = get_batch_logps(ref_outputs.logits.float(), f_labels.cpu()).to(device)

            # 过滤有效样本
            shift_labels = f_labels[..., 1:]
            token_counts = (shift_labels != -100).sum(dim=-1)
            valid_mask = token_counts >= 2
            if valid_mask.sum() == 0:
                if args.debug:
                    print(f"Step {step}: No valid samples, skipping")
                continue

            policy_logps = policy_logps[valid_mask]
            ref_logps    = ref_logps[valid_mask]

            policy_logps = torch.nan_to_num(policy_logps, nan=0.0, posinf=10.0, neginf=-10.0)
            ref_logps    = torch.nan_to_num(ref_logps,    nan=0.0, posinf=10.0, neginf=-10.0)

            if args.debug and step % 5 == 0:
                print(f"Step {step}: policy_logps mean = {policy_logps.mean().item():.4f}, "
                      f"ref_logps mean = {ref_logps.mean().item():.4f}")

            forget_loss = npo_loss(policy_logps, ref_logps, beta=args.beta)

            # =========================
            # retain: 标准 LM loss（关键）
            # =========================
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                policy_retain_outputs = model(
                    r_input_ids,
                    attention_mask=r_mask,
                    use_cache=False
                )
                retain_ce_loss = F.cross_entropy(
                    policy_retain_outputs.logits[..., :-1, :].reshape(-1, policy_retain_outputs.logits.size(-1)),
                    r_labels[..., 1:].reshape(-1),
                    ignore_index=-100
                )

            # 总损失（使用 retain_weight 参数）
            loss = forget_loss + args.retain_weight * retain_ce_loss
            loss = loss / args.grad_acc_steps
            scaler.scale(loss).backward()

            # 梯度累积：达到累积步数才更新
            if (step + 1) % args.grad_acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_history.append(loss.item() * args.grad_acc_steps)  # 记录原始loss

            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_acc_steps:.4f}",
                "forget": f"{forget_loss.item():.4f}",
                "retain": f"{retain_ce_loss.item():.4f}"
            })

        # 保存模型
        save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        plot_loss(loss_history, os.path.join(args.save_dir, f"loss_epoch_{epoch+1}.png"))

if __name__ == "__main__":
    train()