import tensorflow as tf
import numpy as np
from functools import partial

def load_dataset(tfrecords_path, batch_size, augment=False, pretrained=False, same_prob=0.5, diff_prob=0.5, repeat=True, train=True):
    """
	Input:
    tfrecords_path
    batch_size (int)
    same_prob (float): probability of retaining images in same class
    diff_prob (float): probability of retaining images in different class
    train (boolean): train or validation
    repeat (boolean): repeat elements in dataset
    
	Return:
	zipped dataset
	"""
    if augment and train:
        augment_phase = True
    else:
        augment_phase = False
        
    x = load_data_from_tfrecord(tfrecords_path, augment_phase, pretrained=pretrained)
    y = load_data_from_tfrecord(tfrecords_path, augment_phase, pretrained=pretrained)
    
    dataset = tf.data.Dataset.zip((x, y))

    if train:
        filter_func = create_filter_func(same_prob, diff_prob)
        dataset = dataset.filter(filter_func)
        
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
#     While the model is executing training step s, the input pipeline is reading the data for step s+1
    dataset = dataset.prefetch(1)
    return dataset
    
# Derived from ImageNet data
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

def _normalize(image):
    """Normalize the image to zero mean and unit variance."""
    offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
    image -= offset

    scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
    image /= scale
    return image

def load_data_from_tfrecord(filename, augment=False, pretrained=False, shuffle_buffer_size=2000):
    # Step 1: Create a TFRecordDataset as an input pipeline.
#     filename = os.path.join(tfrecord_location, name)
    dataset = tf.data.TFRecordDataset(filename)

    # Step 2: Define a decoder to read and parse data.
    def decode(serialized_example, augment=False, pretrained=False):
        """
        Parses an image and related info from the given `serialized_example`.
        It is used as a map function for `dataset.map`
        """
        # 1. define a parser
        features = tf.io.parse_single_example(
            serialized_example,
            features = {'image/category': tf.FixedLenFeature((), tf.int64, default_value=1),
              'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
              'image/height': tf.FixedLenFeature([], tf.int64),
              'image/width': tf.FixedLenFeature([], tf.int64),
              'image/format': tf.FixedLenFeature((), tf.string, default_value="")
             }
        )

        # 2. Convert the data
        height = tf.cast(features['image/height'], tf.int32)
        width = tf.cast(features['image/width'], tf.int32)
        image = tf.decode_raw(features['image/encoded'], tf.uint8)
        image = tf.cast(image, tf.float32)

        # 3. reshape
        image = tf.reshape(image, [height, width, 3])
   
#         img_b = tf.identity(image)*(1/255.0)
        
        # 4. Augment
        if augment:
            image = tf.image.flip_left_right(image)
        
        # 5: Preprocess the data.
        # Convert `image` from [0, 255] -> [0, 1.0] floats
        if pretrained:
            image = _normalize(image)
        else:
            image = image * (1. / 255)

        return image, features['image/category'], features['image/format']

    # Randomly shuffle the data
    dataset = dataset.shuffle(shuffle_buffer_size)
    decode_fn = partial(decode, augment=augment, pretrained=pretrained)
    # Parse the record into tensors with map. map takes a Python function and applies it to every sample
    dataset = dataset.map(decode_fn, num_parallel_calls=4)  

    return dataset

def get_data(path, batch_size, augment, pretrained, train):
    #import pdb; pdb.set_trace()
    dataset = load_dataset(path, batch_size=batch_size, augment=augment, pretrained=pretrained, train=train)
#     iterator = dataset.make_initializable_iterator()#make_one_shot_iterator()
#     next_elements = iterator.get_next()
#     inputs, target = next_elements
#     return inputs, target, iterator.initializer
    return dataset


def create_filter_func(same_prob, diff_prob):
    def filter_func(left, right):
        _, right_label, _ = left
        _, left_label, _ = right

        label_cond = tf.equal(right_label, left_label)

        different_labels = tf.fill(tf.shape(label_cond), diff_prob)
        same_labels = tf.fill(tf.shape(label_cond), same_prob)

        weights = tf.where(label_cond, same_labels, different_labels)
        random_tensor = tf.random_uniform(shape=tf.shape(weights))

        return weights > random_tensor

    return filter_func
