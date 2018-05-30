#!/bin/bash

CUDA_VISIBLE_DEVICES="0" python restoreCkptReducedClasses_main.py --logtostderr --pipeline_config_path=mytrain/ssd_mobilenetv2_reducedcoco/pipeline.config --train_dir=mytrain/ssd_mobilenetv2_reducedcoco/train/
