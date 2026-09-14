python3 tools/infer.py \
    -c configs/picodet/legacy_model/application/layout_analysis/custom.yaml \
    -o weights='./output/picodet_lcnet_x1_0_layout/best_model' \
    --infer_dir='/workspace/dataset/test' \
    --output_dir='/workspace/test_results_paddle' \
    --draw_threshold=0.5