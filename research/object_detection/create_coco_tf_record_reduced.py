r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2014
        +val2014
        +annotations
            -instances_train2014.json
            -instances_val2014.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python create_coco_tf_record.py --data_dir=/path/to/your/coco/root/directory \
        --set=train \
        --output_filepath=/where/you/want/to/save/pascal.record
        --shuffle_imgs=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging

from utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
FLAGS = flags.FLAGS


def load_coco_dection_dataset(imgs_dir, annotations_filepath, split, shuffle_img = True ):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """

    catIds = [1,3]

    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds(catIds=catIds) # totally 82783 images
    #cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

   
    if shuffle_img:
        shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    totalAnns = np.zeros(len(catIds))
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Reading images: %d / %d "%(index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=catIds)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            totalAnns[catIds.index(ann['category_id'])] += 1
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            #Change car ID from 3 to 2 -Priyabrata
            if ann['category_id'] == 3:
                ann['category_id'] = 2
            labels.append(ann['category_id'])

        img_path = os.path.join(imgs_dir, 'COCO_' + split + '2017_' + img_detail['file_name'])
        img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        coco_data.append(img_info)

    print(totalAnns)
    # exit()
    return coco_data


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

def main(_):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.data_dir, 'images/train2017')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_train2017.json')
        print("Convert coco train file to tf record")
        split = FLAGS.set
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'images/val2017')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_val2017.json')
        print("Convert coco val file to tf record")
        split = FLAGS.set
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data = load_coco_dection_dataset(imgs_dir,annotations_filepath, split, shuffle_img=FLAGS.shuffle_imgs)
    total_imgs = len(coco_data)
    # write coco data to tf record
    with tf.python_io.TFRecordWriter(FLAGS.output_filepath) as tfrecord_writer:
        for index, img_data in enumerate(coco_data):
            if index % 100 == 0:
                print("Converting images: %d / %d" % (index, total_imgs))
            example = dict_to_coco_example(img_data)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
    tf.app.run()
