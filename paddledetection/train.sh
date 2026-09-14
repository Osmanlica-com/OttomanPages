export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py \
    -c configs/picodet/legacy_model/application/layout_analysis/custom.yaml \
    --eval