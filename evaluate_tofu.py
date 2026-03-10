import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# 复用数据加载函数
def load_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def extract_texts(raw_data):
    texts = []
    for item in raw_data:
        if isinstance(item, dict):
            if 'question' in item and 'answer' in item:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            elif 'text' in item:
                text = item['text']
            elif 'prompt' in item and 'completion' in item:
                text = f"{item['prompt']} {item['completion']}"
            else:
                text = ' '.join([str(v) for v in item.values() if isinstance(v, str)])
        elif isinstance(item, str):
            text = item
        else:
            continue
        if len(text) > 10:
            texts.append(text)
    return texts

class TOFUEvalDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        raw_data = load_jsonl_file(filepath)
        self.texts = extract_texts(raw_data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding=False
        )
        return enc.input_ids[0], enc.attention_mask[0]

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Evaluating"):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum',
                ignore_index=-100
            )
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()
            if total_tokens == 0:  # 在循环内添加
                print("Sample input_ids:", input_ids[0])
                print("Sample labels:", labels[0])
                print("Sample attention_mask:", attention_mask[0])
                print("Shift labels (first 20):", shift_labels[0, :20])
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="训练后模型保存路径")
    parser.add_argument("--data_dir", type=str, default="./tofu_data", help="TOFU 数据目录")
    parser.add_argument("--forget_pct", type=int, default=10, help="遗忘百分比")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None
    ).to(device)

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name}")
        if torch.isinf(param).any():
            print(f"Inf in {name}")

    # 定义 collate_fn（内部函数，可访问 tokenizer）
    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        return input_ids, attention_mask

    # 定义要评估的数据集文件
    eval_sets = {
        "forget": f"{args.data_dir}/forget{args.forget_pct:02d}.json",
        "retain": f"{args.data_dir}/retain{95}.json",  # 假设 retain95
        "holdout": f"{args.data_dir}/holdout{args.forget_pct:02d}.json",
        "real_authors": f"{args.data_dir}/real_authors.json",
        "world_facts": f"{args.data_dir}/world_facts.json"  # 如果有
    }

    results = {}
    for name, filepath in eval_sets.items():
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
        print(f"\nEvaluating on {name} set: {filepath}")
        dataset = TOFUEvalDataset(filepath, tokenizer, args.max_length)
        if len(dataset) == 0:
            print(f"  No valid samples in {name} set.")
            continue
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        avg_loss, ppl = compute_perplexity(model, dataloader, device)
        results[name] = {
            "loss": avg_loss,
            "perplexity": ppl,
            "num_samples": len(dataset)
        }
        print(f"  Loss: {avg_loss:.4f}, Perplexity: {ppl:.4f}")

    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()