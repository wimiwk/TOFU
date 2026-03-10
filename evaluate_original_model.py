import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# 复用之前的数据加载函数
def load_jsonl_file(filepath):
    """加载JSONL格式的文件（每行一个JSON对象）"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line in {filepath}: {e}")
                    continue
    return data

def extract_text_from_item(item):
    """从TOFU数据项中提取文本"""
    if isinstance(item, dict):
        if 'question' in item and 'answer' in item:
            return f"Question: {item['question']}\nAnswer: {item['answer']}"
        elif 'text' in item:
            return item['text']
        elif 'prompt' in item and 'completion' in item:
            return f"{item['prompt']} {item['completion']}"
        else:
            # 回退：将所有字符串值拼接
            return ' '.join([str(v) for v in item.values() if isinstance(v, str)])
    elif isinstance(item, str):
        return item
    else:
        return ""

class TextDataset(Dataset):
    """简单的文本数据集，用于评估"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
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
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = enc['input_ids'][0]
        attention_mask = enc['attention_mask'][0]
        # 标签与输入相同，但将padding部分设为-100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

@torch.no_grad()
def evaluate_split(model, tokenizer, split_name, file_path, batch_size=8, max_length=512, device='cuda'):
    """评估单个数据分片"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping {split_name}")
        return None

    raw_data = load_jsonl_file(file_path)
    texts = [extract_text_from_item(item) for item in raw_data if extract_text_from_item(item)]
    if not texts:
        print(f"Warning: No valid texts in {file_path}")
        return None

    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算交叉熵损失（不reduction，保留每个token的loss）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 忽略padding位置的损失
        mask = (shift_labels.view(-1) != -100)
        loss = loss[mask]
        total_loss += loss.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss) if avg_loss != float('inf') else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_samples': len(texts)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate original model on TOFU splits")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-1_5", help="Hugging Face model ID")
    parser.add_argument("--data_dir", type=str, default="./tofu_data", help="TOFU data directory")
    parser.add_argument("--forget_pct", type=int, default=10, help="Forget set percentage (1,5,10,100)")
    parser.add_argument("--retain_pct", type=int, default=95, help="Retain set percentage (90,95,99)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output_file", type=str, default="original_model_eval.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading tokenizer and model from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        trust_remote_code=True
    ).to(args.device)
    model.eval()

    # 定义要评估的 splits 及其对应的文件路径
    splits = [
        ('forget', f"{args.data_dir}/forget{args.forget_pct:02d}.json" if args.forget_pct != 100 else f"{args.data_dir}/full.json"),
        ('retain', f"{args.data_dir}/retain{args.retain_pct}.json"),
        ('holdout', f"{args.data_dir}/holdout{args.forget_pct:02d}.json"),
        ('real_authors', f"{args.data_dir}/real_authors.json"),
        ('world_facts', f"{args.data_dir}/world_facts.json")
    ]

    results = {}
    for name, file_path in splits:
        print(f"\n--- Evaluating {name} ---")
        res = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            split_name=name,
            file_path=file_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device
        )
        if res is not None:
            results[name] = res

    # 输出结果
    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))

    # 保存到文件
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()