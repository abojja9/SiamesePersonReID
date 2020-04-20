import glob
import argparse
import os
import random
import numpy as np
import sys
import PIL
from pathlib import Path
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data', help='path of the dataset')
parser.add_argument('--tf_record_dir', dest='tf_record_dir', default='./tf_record_dir', help='path of the dataset')
parser.add_argument('--subset_data', dest='subset_data', type=bool, default=False, help='if subset_data is true use 1000 sample to create the data: 1: true, 0: false')
args = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_filenames(data_root):
    directories, categories, filenames = [], [], []
    if os.path.exists(data_root):
        for filename in os.listdir(data_root):
            path = os.path.join(data_root, filename)
            if os.path.isdir(path):
                directories.append(path)
                categories.append(filename)
        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                filenames.append(path)
    return filenames, categories

def createDataRecord(out_filename, fnames, category2ids):
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(fnames)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print(f'{out_filename}: {i}/{len(fnames)}')
            sys.stdout.flush()
        # Load the image
        img = np.array(PIL.Image.open(fnames[i]))

        if img is None:
            continue
        
        height = img.shape[0]
        width = img.shape[1]
        
        cat_name = os.path.basename(os.path.dirname(fnames[i]))
        cat_id = category2ids[cat_name]
        
        # Create a feature
        feature = {
            'image/encoded': _bytes_feature(img.tostring()),
            'image/format': _bytes_feature( b'jpg'),
            'image/category': _int64_feature(cat_id),
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width)
        }
        
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()
    
    
def write_label_file(labels_to_categories, dataset_dir,
                     filename='labels.txt'):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_categories:
            class_name = labels_to_categories[label]
            f.write('%d:%s\n' % (label, class_name))

def write_data_summary(num_validation, num_dataset, dataset_dir, filename='data_summary.txt'):
    """Writes a file with the number of validation and dataset images."""
    data_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(data_filename, 'w') as f:
        f.write('%d\n%d' % (num_validation, num_dataset))
    

    
def main():
    print ("[*] In TF records file")
    dataset_dir = args.dataset_dir
    train_data = 'bbox_train'
    test_data = 'bbox_test'
    validation_size = 0.2
  
    tf_record_dir = Path(args.tf_record_dir)
    if not tf_record_dir.exists():
        train_data_root = os.path.join(dataset_dir, train_data)
        test_data_root = os.path.join(dataset_dir, test_data)
        filenames, categories = get_filenames(train_data_root)
        test_fnames, test_categories = get_filenames(test_data_root)
    
        # Use subset data for debugging purpose
        if args.subset_data:
            print ("[*] Using subset of data")
            filenames = filenames[:100000]
    
        validation_size = 0.2
        train_category2ids = dict(zip(categories, range(len(categories))))
        test_category2ids = dict(zip(test_categories, range(len(test_categories))))
        val_size = int(validation_size * len(filenames))
    
        random.seed(32)
        random.shuffle(filenames)
        train_fnames = filenames[val_size:]
        val_fnames = filenames[:val_size]

        print ("[*] Creating tf_records_data directory")
        tf_record_dir.mkdir(parents=True)
        createDataRecord(str(tf_record_dir/'train.tfrecords'), train_fnames, train_category2ids)
        createDataRecord(str(tf_record_dir/'val.tfrecords'), val_fnames, train_category2ids)
        createDataRecord(str(tf_record_dir/'test.tfrecords'), test_fnames, test_category2ids)
        
        # Finally, write the labels file:
        labels_to_categories = dict(zip(range(len(categories)), categories))
        write_label_file(labels_to_categories, tf_record_dir)
        write_data_summary(val_size, len(filenames), tf_record_dir)
        print ("[*] Finished creating TF records")
    else:
        print ("TFRecords already created!")
        

if __name__ == '__main__':
    main()
