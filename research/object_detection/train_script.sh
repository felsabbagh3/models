#!/bin/bash

#CUDA_VISIBLE_DEVICES="0" python train.py --logtostderr --pipeline_config_path=mytrain/ssd_mobilenetv2_reducedcoco/pipeline.config --train_dir=mytrain/ssd_mobilenetv2_reducedcoco/train/
CUDA_VISIBLE_DEVICES="1" python train.py --logtostderr --pipeline_config_path=mytrain/rfcn_resnet101_coco_2018_01_28/pipeline.config --train_dir=mytrain/rfcn_resnet101_coco_2018_01_28/train/
#/home/bmudassar3/work/models/research/object_detection/mytrain/rfcn_resnet101_coco_2018_01_28