python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    tools/test_net.py \
    --config-file $2 \
    OUTPUT_DIR $3 \
    MODEL.WEIGHT $4
