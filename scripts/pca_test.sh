CUDA_VISIBLE_DEVICES=0 python main.py \
    --model MLP \
    --log_file pca.log \
    --result_file pca.csv \
    --use_pca \
    --pca_dim 500

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model MLP \
    --log_file pca.log \
    --result_file pca.csv \
    --use_pca \
    --pca_dim 1000

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model MLP \
    --log_file pca.log \
    --result_file pca.csv \
    --use_pca \
    --pca_dim 2000

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model MLP \
    --log_file pca.log \
    --result_file pca.csv \
    --use_pca \
    --pca_dim 5000
    