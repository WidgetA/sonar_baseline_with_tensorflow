# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import os
import sys


from lxml import etree
import tensorflow as tf

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.dirname(path)

from utils import dataset_util
from utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', True, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       filename):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

    Returns:
    example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    full_path = dataset_directory
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    if 'object' in data:
        for obj in data['object']:
            width = float(obj['bndbox']['xmax']) - float(obj['bndbox']['xmin'])
            height = float(obj['bndbox']['ymax']) - float(obj['bndbox']['ymin'])
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': dataset_util.bytes_feature(
          filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
          filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    return example


def main(_):
    years = [FLAGS.year]

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path.encode('utf8'))
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    for year in years:
        logging.info('Reading from PASCAL %s dataset.', year)
        examples_path = FLAGS.set
        annotations_dir = FLAGS.annotations_dir
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data_path = os.path.join(FLAGS.data_dir, example + '.jpg')
            logging.info(data_path)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            tf_example = dict_to_tf_example(data, data_path, label_map_dict, example)
            writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()

