import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util


parser = argparse.ArgumentParser()
parser.add_argument("-step", help="Specify the training step")
parser.add_argument("-path", help="Specify the path of images")
args = parser.parse_args()

if 'frozen_inference_graph.pb' not in os.listdir('model_file/frozen_pb'):
    os.system('python object_detection/export_inference_graph.py ' +
              '--pipeline_config_path=model_file/train/pipeline.config ' +
              f'--trained_checkpoint_prefix=model_file/train/model.ckpt-{args.step} ' +
              '--output_directory=model_file/frozen_pb')
else:
    pass


model_path = 'model_file/frozen_pb/frozen_inference_graph.pb'
label_path = 'dataset/pascal_label_map.pbtxt'
img_path = [os.path.join(args.path, x) for x in os.listdir(args.path)]

num_classes = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        with open('submit.csv', 'w+') as f:
            f.write('name,image_id,confidence,xmin,ymin,xmax,ymax\n')
            for image_path in img_path:
                image_np = cv2.imread(image_path)
                width = image_np.shape[1]
                height = image_np.shape[0]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                for box, score, nclass in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                    ymin = int(box[0] * height)
                    xmin = int(box[1] * width)
                    ymax = int(box[2] * height)
                    xmax = int(box[3] * width)
                    if score > 0.15 and (xmax - xmin) * (ymax - ymin) < 0.6 * height*width:
                        f.write('target,' + str(os.path.basename(image_path))[:-4] +
                                '.xml,' +
                                str(score) +
                                ',' +
                                str(xmin) +
                                ',' +
                                str(ymin) +
                                ',' +
                                str(xmax) +
                                ',' +
                                str(ymax)
                                + '\n')
                    else:
                        continue
            f.close()


