import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np

def load_jsonl_file(filepath):
    """加载JSONL格式的文件（每行一个JSON对象）"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line in {filepath}: {e}")
                    continue
    return data

def convert_raw_data_to_model_format(data, tokenizer, max_length):
    """将原始文本转换为模型输入格式"""
    unlearn_input = tokenizer(
        data, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    return unlearn_input

class TOFUDataset(Dataset):
    """
    TOFU数据集加载器
    根据TOFU数据集的文件结构进行适配
    """
    def __init__(self, tokenizer, data_dir="./tofu_data", forget_percentage=5, 
                 retain_percentage=95, coreset_percentage=1.0, seed=42, 
                 token_len=512, min_len=50, mode="train"):
        """
        Args:
            tokenizer: 分词器
            data_dir: TOFU数据目录
            forget_percentage: 遗忘集百分比 (1, 5, 10, 100)
            retain_percentage: 保留集百分比 (90, 95, 99)
            coreset_percentage: 核集选择比例 (0-1.0)
            seed: 随机种子
            token_len: 最大token长度
            min_len: 最小文本长度
            mode: 模式 ('train' 或 'eval')
        """
        super(TOFUDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.token_len = token_len
        self.min_len = min_len
        self.mode = mode
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 根据百分比确定文件名
        if forget_percentage == 100:
            forget_file = f"{data_dir}/full.json"
        else:
            forget_file = f"{data_dir}/forget{forget_percentage:02d}.json"
        
        retain_file = f"{data_dir}/retain{retain_percentage}.json"
        
        print(f"Loading forget data from: {forget_file}")
        print(f"Loading retain data from: {retain_file}")
        
        # 使用JSONL格式加载数据
        forget_data_raw = load_jsonl_file(forget_file)
        retain_data_raw = load_jsonl_file(retain_file)
        
        # 提取文本数据
        self.forget_data = self._extract_texts(forget_data_raw)
        self.retain_data = self._extract_texts(retain_data_raw)
        
        # 应用核集选择（如果是训练模式）
        if mode == "train" and coreset_percentage < 1.0:
            self._apply_coreset_selection(coreset_percentage, seed)
        
        # 对于评估模式，我们可能还需要其他文件
        if mode == "eval":
            # 加载holdout数据用于评估
            holdout_file = f"{data_dir}/holdout{forget_percentage:02d}.json"
            try:
                holdout_data_raw = load_jsonl_file(holdout_file)
                self.holdout_data = self._extract_texts(holdout_data_raw)
            except FileNotFoundError:
                print(f"Holdout file not found: {holdout_file}")
                self.holdout_data = []
            
            # 加载真实作者数据用于保留知识评估
            real_authors_file = f"{data_dir}/real_authors.json"
            try:
                real_authors_raw = load_jsonl_file(real_authors_file)
                self.real_authors_data = self._extract_texts(real_authors_raw)
            except FileNotFoundError:
                print(f"Real authors file not found: {real_authors_file}")
                self.real_authors_data = []
        
        print(f"TOFU {mode} dataset loaded: forget={len(self.forget_data)}, "
              f"retain={len(self.retain_data)}")
    
    def _extract_texts(self, raw_data):
        """从原始数据中提取文本"""
        texts = []
        for item in raw_data:
            # TOFU数据集格式：每个项目是一个包含question和answer的字典
            if isinstance(item, dict):
                if 'question' in item and 'answer' in item:
                    # 对于TOFU数据集，我们使用问题+答案作为文本
                    text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                elif 'text' in item:
                    text = item['text']
                elif 'prompt' in item and 'completion' in item:
                    text = f"{item['prompt']} {item['completion']}"
                else:
                    # 尝试使用所有字符串值
                    text = ' '.join([str(v) for v in item.values() if isinstance(v, str)])
            elif isinstance(item, str):
                text = item
            else:
                continue
            
            if len(text) > self.min_len:
                texts.append(text)
        
        return texts
    
    def _apply_coreset_selection(self, percentage, seed):
        """应用核集选择"""
        if percentage >= 1.0 or len(self.forget_data) == 0:
            return
        
        num_samples = int(len(self.forget_data) * percentage)
        indices = random.sample(range(len(self.forget_data)), num_samples)
        self.forget_data = [self.forget_data[i] for i in indices]
        
        print(f"Applied coreset selection: selected {num_samples} samples "
              f"({percentage*100:.1f}%) from forget set")
    
    def __len__(self):
        """返回数据集长度（遗忘集大小）"""
        return len(self.forget_data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取遗忘数据
        forget_text = self.forget_data[idx]
        
        # 获取保留数据（循环使用）
        retain_idx = idx % len(self.retain_data)
        retain_text = self.retain_data[retain_idx]
        
        # 转换为模型输入格式
        forget_encoded = convert_raw_data_to_model_format(
            forget_text, self.tokenizer, self.token_len
        )
        retain_encoded = convert_raw_data_to_model_format(
            retain_text, self.tokenizer, self.token_len
        )
        
        return [forget_encoded, retain_encoded], idx


class TOFUIndexedDataset(TOFUDataset):
    """支持预计算核集索引的TOFU数据集"""
    def __init__(self, tokenizer, data_dir="./tofu_data", forget_percentage=5,
                 retain_percentage=95, index_file=None, token_len=512, 
                 min_len=50, mode="train"):
        """
        Args:
            index_file: 预计算的核集索引文件
        """
        # 先调用父类初始化，但不应用核集选择
        super().__init__(
            tokenizer=tokenizer,
            data_dir=data_dir,
            forget_percentage=forget_percentage,
            retain_percentage=retain_percentage,
            coreset_percentage=1.0,  # 不应用随机选择
            seed=42,
            token_len=token_len,
            min_len=min_len,
            mode=mode
        )
        
        # 如果有索引文件，应用预计算的核集
        if index_file and mode == "train":
            self._apply_precomputed_coreset(index_file)
    
    def _apply_precomputed_coreset(self, index_file):
        """应用预计算的核集索引"""
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                indices = json.load(f)
            
            # 保存完整的遗忘数据
            full_forget_data = self.forget_data.copy()
            
            # 应用索引选择
            self.forget_data = [full_forget_data[i] for i in indices]
            
            print(f"Applied precomputed coreset from {index_file}: "
                  f"selected {len(self.forget_data)} samples")
        except Exception as e:
            print(f"Error loading precomputed coreset: {e}")
            print("Using full forget set instead")


def tofu_custom_collator(samples):
    """TOFU数据集的collator函数"""
    idxs = [s[1] for s in samples]
    
    # 分离遗忘数据和保留数据
    forget_data_list = [s[0][0] for s in samples]
    retain_data_list = [s[0][1] for s in samples]
    
    # 堆叠输入ID
    forget_input_ids = torch.stack([f['input_ids'][0] for f in forget_data_list])
    forget_attention_mask = torch.stack([f['attention_mask'][0] for f in forget_data_list])
    
    retain_input_ids = torch.stack([r['input_ids'][0] for r in retain_data_list])
    retain_attention_mask = torch.stack([r['attention_mask'][0] for r in retain_data_list])
    
    # 返回格式与RMU代码兼容
    return (forget_input_ids, forget_attention_mask), \
           (retain_input_ids, retain_attention_mask), \
           idxs


def tofu_custom_collator_forget(samples):
    """用于NPO等需要标签的collator"""
    idxs = [s[1] for s in samples]
    
    forget_data_list = [s[0][0] for s in samples]
    retain_data_list = [s[0][1] for s in samples]
    
    # 遗忘数据
    forget_input_ids = torch.stack([f['input_ids'][0] for f in forget_data_list])
    forget_attention_mask = torch.stack([f['attention_mask'][0] for f in forget_data_list])
    forget_labels = forget_input_ids.clone()  # 使用输入ID作为标签
    forget_labels[forget_attention_mask == 0] = -100   # padding 位置设为 -100
    
    # 保留数据
    retain_input_ids = torch.stack([r['input_ids'][0] for r in retain_data_list])
    retain_attention_mask = torch.stack([r['attention_mask'][0] for r in retain_data_list])
    retain_labels = retain_input_ids.clone()
    
    return (forget_input_ids, forget_labels, forget_attention_mask), \
           (retain_input_ids, retain_labels, retain_attention_mask), \
           idxs