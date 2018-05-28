#!/bin/bash

# ./run_export_inference_graph.sh 
# object_detection/checkpoints/faster_rcnn_inception_v2_coco_gaussian_avgfilter_similarity1e-5/pipeline.config 
# object_detection/checkpoints/faster_rcnn_inception_v2_coco_gaussian_avgfilter_similarity1e-5/model.ckpt-800000
# object_detection/checkpoints/faster_rcnn_inception_v2_coco_gaussian_avgfilter_similarity1e-5/


PIPELINE_CONFIG_PATH=$1
TRAIN_PATH=$2
OUTPUT_PATH=$3

# From tensorflow/models/research/
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory ${OUTPUT_PATH}
