from os.path import dirname,exists,join
from datetime import datetime
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

model_dir = 'data/'
image_dir = 'images/'
graph_file = 'classify_image_graph_def.pb'

def create_graph():
    with tf.gfile.FastGFile(join(model_dir,graph_file),'rb') as graph_data:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_data.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(image):
    nb_features = 2048
    feature = np.empty(nb_features,)

    try:
        with tf.Session() as sess:
            next_to_last_tensor = sess.get_tensor_by_name('pool3:0')
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0':image_data})
            feature = np.squeeze(predictions)
    except Exception as e:
        print e
    return feature

create_graph()
