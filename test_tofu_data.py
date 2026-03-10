import json
from tofu_data_module import load_jsonl_file, TOFUDataset
from transformers import AutoTokenizer
import os

def test_jsonl_loading():
    """测试JSONL文件加载"""
    print("="*80)
    print("Testing JSONL file loading...")
    
    # 测试加载forget文件
    forget_data = load_jsonl_file("./tofu_data/forget05.json")
    print(f"Loaded {len(forget_data)} items from forget05.json")
    
    # 显示前几个项目
    print("\nFirst 3 forget samples:")
    for i, item in enumerate(forget_data[:3]):
        print(f"\nItem {i+1}:")
        print(f"  Question: {item.get('question', 'No question')[:50]}...")
        print(f"  Answer: {item.get('answer', 'No answer')[:50]}...")
    
    # 测试加载retain文件
    retain_data = load_jsonl_file("./tofu_data/retain95.json")
    print(f"\nLoaded {len(retain_data)} items from retain95.json")
    
    # 显示前几个项目
    print("\nFirst 3 retain samples:")
    for i, item in enumerate(retain_data[:3]):
        print(f"\nItem {i+1}:")
        print(f"  Question: {item.get('question', 'No question')[:50]}...")
        print(f"  Answer: {item.get('answer', 'No answer')[:50]}...")
    
    return forget_data, retain_data

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "="*80)
    print("Testing dataset loading...")
    
    # 检查文件是否存在
    if not os.path.exists("./tofu_data/forget05.json"):
        print("Error: forget05.json not found!")
        return None
    
    if not os.path.exists("./tofu_data/retain95.json"):
        print("Error: retain95.json not found!")
        return None
    
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # 使用简单tokenizer作为fallback
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    try:
        dataset = TOFUDataset(
            tokenizer=tokenizer,
            data_dir="./tofu_data",
            forget_percentage=5,
            retain_percentage=95,
            coreset_percentage=0.1,  # 10%核集
            seed=42,
            token_len=128,
            min_len=10,
            mode="train"
        )
        
        print(f"Dataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        # 获取一个样本
        sample, idx = dataset[0]
        forget_encoded, retain_encoded = sample
        
        print(f"\nSample 0:")
        print(f"  Forget input shape: {forget_encoded['input_ids'].shape}")
        print(f"  Retain input shape: {retain_encoded['input_ids'].shape}")
        
        # 解码查看文本
        forget_text = tokenizer.decode(forget_encoded['input_ids'][0], skip_special_tokens=True)
        retain_text = tokenizer.decode(retain_encoded['input_ids'][0], skip_special_tokens=True)
        
        print(f"\nForget text (first 100 chars): {forget_text[:100]}...")
        print(f"Retain text (first 100 chars): {retain_text[:100]}...")
        
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_coreset_indices():
    """测试核集索引创建"""
    print("\n" + "="*80)
    print("Testing coreset indices creation...")
    
    try:
        from create_tofu_coreset import create_random_coreset_indices
        
        # 创建核集索引
        index_file = create_random_coreset_indices(
            data_dir="./tofu_data",
            forget_percentage=5,
            coreset_percentage=5.0,
            seed=42
        )
        
        # 加载并显示索引
        with open(index_file, 'r') as f:
            indices = json.load(f)
        
        print(f"Created {len(indices)} indices:")
        print(f"First 10 indices: {indices[:10]}")
        
        return indices
    except Exception as e:
        print(f"Error creating coreset indices: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("Starting TOFU dataset tests...")
    
    # 测试1: JSONL加载
    forget_data, retain_data = test_jsonl_loading()
    
    # 测试2: 数据集加载
    dataset = test_dataset_loading()
    
    # 测试3: 核集索引
    indices = test_coreset_indices()
    
    print("\n" + "="*80)
    print("Test Summary:")
    print(f"  JSONL loading: {'✓' if forget_data and retain_data else '✗'}")
    print(f"  Dataset creation: {'✓' if dataset else '✗'}")
    print(f"  Coreset indices: {'✓' if indices else '✗'}")
    print("="*80)

if __name__ == "__main__":
    main()