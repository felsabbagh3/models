#!/bin/bash

CUDA_VISIBLE_DEVICES="1" python train.py --logtostderr --pipeline_config_path=/home/felsabbagh3/Desktop/models/research/object_detection/mytrain/ssd_mobilenetv2_reducedcoco/pipeline.config --train_dir=mytrain/ssd_mobilenetv2_reducedcoco/train/
