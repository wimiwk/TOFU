import json
import random
import os
import argparse
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, AutoModelForCausalLM

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
                    print(f"Error parsing JSON line: {line}")
                    print(f"Error: {e}")
                    continue
    return data

def extract_embeddings(texts, model_id='microsoft/phi-1_5', batch_size=8, device=None):
    """提取文本的embedding (mean pooling of last hidden states)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {model_id} for embedding extraction on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            mean_pool = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(mean_pool.cpu().numpy())
    return np.vstack(embeddings)

def kcenter_greedy(embeddings, k):
    """最远点采样 (K-Center Greedy)，返回Python int列表"""
    n = embeddings.shape[0]
    distances = np.full(n, np.inf)
    selected = []
    # 随机选择第一个点
    current = np.random.randint(n)
    selected.append(int(current))   # 转换为Python int
    dist_to_current = euclidean_distances(embeddings, embeddings[current].reshape(1, -1)).flatten()
    distances = np.minimum(distances, dist_to_current)

    for _ in range(1, k):
        current = np.argmax(distances)
        selected.append(int(current))
        dist_to_current = euclidean_distances(embeddings, embeddings[current].reshape(1, -1)).flatten()
        distances = np.minimum(distances, dist_to_current)
    return selected

def create_random_coreset_indices(data_dir, forget_percentage, coreset_percentage,
                                  method='random', seed=42, model_id='microsoft/phi-1_5',
                                  output_dir="./tofu_data/indices"):
    """
    为TOFU数据集创建核集索引

    Args:
        data_dir: TOFU数据目录
        forget_percentage: 遗忘集百分比 (1, 5, 10, 100)
        coreset_percentage: 核集选择比例 (1, 5, 10, 100)
        method: 采样方法 ('random' 或 'embedding_kcenter')
        seed: 随机种子
        model_id: 用于提取embedding的模型ID
        output_dir: 输出目录
    """
    # 确定文件名
    if forget_percentage == 100:
        forget_file = f"{data_dir}/full.json"
    else:
        forget_file = f"{data_dir}/forget{forget_percentage:02d}.json"

    print(f"Loading forget data from: {forget_file}")
    forget_data = load_jsonl_file(forget_file)

    total_samples = len(forget_data)
    selected_samples = int(total_samples * (coreset_percentage / 100.0))
    print(f"Total samples in forget set: {total_samples}")
    print(f"Selecting {selected_samples} samples ({coreset_percentage}%) using method '{method}'")

    if method == 'random':
        random.seed(seed)
        indices = random.sample(range(total_samples), selected_samples)
    elif method == 'embedding_kcenter':
        # 提取文本
        texts = []
        for item in forget_data:
            if 'question' in item and 'answer' in item:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            else:
                # 后备方案：将字典转为字符串
                text = ' '.join([str(v) for v in item.values() if isinstance(v, str)])
            texts.append(text)

        print("Extracting embeddings...")
        embeddings = extract_embeddings(texts, model_id=model_id)
        print("Running k-center greedy...")
        np.random.seed(seed)  # 保证可复现
        indices = kcenter_greedy(embeddings, selected_samples)
        # 确保所有索引为Python int（已由函数内部转换，这里再确认一次）
        indices = [int(idx) for idx in indices]
    else:
        raise ValueError(f"Unknown method: {method}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存索引（文件名包含方法名）
    output_file = f"{output_dir}/tofu_forget{forget_percentage:02d}_{method}_{coreset_percentage}_{seed}.json"
    with open(output_file, 'w') as f:
        json.dump(indices, f)

    print(f"Created coreset indices:")
    print(f"  Total samples: {total_samples}")
    print(f"  Selected samples: {selected_samples} ({coreset_percentage}%)")
    print(f"  Saved to: {output_file}")

    # 预览
    print("\nFirst 5 selected samples preview:")
    for i, idx in enumerate(indices[:5]):
        if idx < len(forget_data):
            sample = forget_data[idx]
            question = sample.get('question', 'No question')
            print(f"  {i+1}. Index {idx}: {question[:80]}...")

    return output_file

def create_coreset_for_all_configs(data_dir, method='random', seeds=[42, 123, 456]):
    """为所有配置创建核集索引"""
    configs = [
        (1, 1), (1, 5), (1, 10),
        (5, 1), (5, 5), (5, 10),
        (10, 1), (10, 5), (10, 10),
        (100, 1), (100, 5), (100, 10),
    ]

    for forget_pct, coreset_pct in configs:
        for seed in seeds:
            create_random_coreset_indices(
                data_dir=data_dir,
                forget_percentage=forget_pct,
                coreset_percentage=coreset_pct,
                method=method,
                seed=seed
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create coreset indices for TOFU dataset")
    parser.add_argument("--data_dir", type=str, default="./tofu_data", help="TOFU data directory")
    parser.add_argument("--forget_pct", type=int, default=5, help="Forget set percentage (1, 5, 10, 100)")
    parser.add_argument("--coreset_pct", type=float, default=5.0, help="Coreset percentage")
    parser.add_argument("--method", type=str, default="random", choices=["random", "embedding_kcenter"],
                        help="Coreset selection method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-1_5", help="Model for embedding extraction (used with embedding_kcenter)")
    parser.add_argument("--all", action="store_true", help="Create indices for all configurations")

    args = parser.parse_args()

    # 检查文件是否存在
    if args.forget_pct == 100:
        file_path = f"{args.data_dir}/full.json"
    else:
        file_path = f"{args.data_dir}/forget{args.forget_pct:02d}.json"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        print("Please check the data directory and forget percentage.")
        exit(1)

    if args.all:
        create_coreset_for_all_configs(args.data_dir, method=args.method)
    else:
        create_random_coreset_indices(
            data_dir=args.data_dir,
            forget_percentage=args.forget_pct,
            coreset_percentage=args.coreset_pct,
            method=args.method,
            seed=args.seed,
            model_id=args.model_id
        )