CUDA_VISIBLE_DEVICES=0 python main.py \
    --model MLP \
    --preprocess \
    --log_file val_acc.log \
    --result_file val_acc.csv \
    --val_when_train
    # --save_model \
    # --use_pca \
    # --pca_dim 500 \
    # --load_pca
    