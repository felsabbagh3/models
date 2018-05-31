#!/bin/bash

CUDA_VISIBLE_DEVICES="1" python eval.py --logtostderr --pipeline_config_path=mytrain/ssd_mobilenetv2_reducedcoco/pipeline.config --checkpoint_dir=mytrain/ssd_mobilenetv2_reducedcoco/train/ --eval_dir=mytrain/ssd_mobilenetv2_reducedcoco/eval/
