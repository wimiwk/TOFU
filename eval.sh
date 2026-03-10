for epoch in 1 2 3 4 5 6 7 8 9 10; do
  python evaluate_tofu.py \
    --model_path ../autodl-tmp/models/phi_npo_tofu_embedding_kcenter/epoch_${epoch} \
    --data_dir ./tofu_data \
    --forget_pct 10 \
    --batch_size 4 \
    --output_file eval_epoch_${epoch}.json
done
