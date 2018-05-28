#!/bin/bash


network="ssd_mobilenetv2_reducedcoco"
command="./run_export_inference_graph.sh mytrain/"$network"/pipeline.config mytrain/"$network"/train/model.ckpt-137436 mytrain/"$network"/frozen_graph/ True mytrain/"$network"/train/gradients.gz"
eval $command

#./run_export_inference_graph.sh ../../models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config ../../models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt ../../models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28

#/home/bmudassar3/work/CAMEL/models/tensorflow/ssd_mobilenetv1_reducedcoco_Burhan