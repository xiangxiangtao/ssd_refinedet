#!/bin/bash
python eval_refinedet.py --image_size 320 --weight_path "weights/refinedet_composite18.1_epoch2.pth"

# python eval_refinedet.py --image_size 320 --weight_folder "checkpoints/refinedet/weight_refinedet_composite18.1"