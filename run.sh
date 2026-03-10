#!/bin/bash
# 使用 Hugging Face 镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 训练参数
MODEL_ID="microsoft/phi-1_5"
FORGET_PCT=10
CORESET_PCT=10.0
CORESET_METHOD="embedding_kcenter"   # 可选 random 或 embedding_kcenter
BETA=0.4
LR=2.5e-5
EPOCHS=10
SEED=42
BATCH_SIZE=1                          # 根据显存调整
GRAD_ACC_STEPS=16                      # 梯度累积步数，增大以节省显存
OPTIMIZER="adamw"                  # 使用8-bit AdamW (需安装bitsandbytes)
GRADIENT_CHECKPOINTING="--gradient_checkpointing"  # 启用梯度检查点

echo "🚀 启动 Phi-1.5B NPO 训练 (NVIDIA GPU) with method: ${CORESET_METHOD}"

python forget_npo_tofu_phi.py \
  --model_id ${MODEL_ID} \
  --forget_pct ${FORGET_PCT} \
  --coreset_pct ${CORESET_PCT} \
  --coreset_method ${CORESET_METHOD} \
  --seed ${SEED} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --beta ${BETA} \
  --retain_weight 1.4 \
  --epochs ${EPOCHS} \
  --mixed_precision \
  --grad_acc_steps ${GRAD_ACC_STEPS} \
  --optimizer ${OPTIMIZER} \
  ${GRADIENT_CHECKPOINTING} \
  --save_dir "../autodl-tmp/models/phi_npo_tofu_${CORESET_METHOD}"

echo "✅ 训练完成。"