# Metric Learning for Person Re-identification
This repo contains the training and evaluation framework for the Person Re Identification task using Siamese Network on MARS dataset. The code is implemented in Tensorflow 1.15.



## Installation
Create a python virtual environment and install the dependancies.

```
pip3 install -r requirements.txt
```

## Download data

Download the MARS dataset from here:
* [MARS](https://drive.google.com/drive/folders/0B6tjyrV1YrHeMVV2UFFXQld6X1E?amp%3Busp=sharing) 

- The folder contains train and test sets in zip format. Unzip the data to the data directory:

```plain
└── data
       ├── bbox_train    
       |   ├── 0001
       |        ....jpg
       |        ....jpg
       |
       └── bbox_test     
           ├── 0000 
                ....jpg
                ....jpg

```

## Training 

First, the code creates the TFRecord files for training, validation and test sets for efficient data feeding to the model from the dataset directory and trains the model.

```
python3 train_net.py /path/to/data/
```

The TF records can be created seperately using the following commnad.
```
python3 data_tfrecord.py --dataset_dir=./data --tf_record_dir=./tf_record_dir
```

The model creates checkpoints and logs to the folders ./checkpoint ./log.

Optionally, we can Use the script scripts/job_contrastive.sh to run the training. Please change the path to the virtual environment used.

```
sh scripts/job_contrastive.sh
```

Training evolution can be viewed on Tensorboard by running
```
tensorboard --logdir ./log
```

## Testing 

```
python3 test_net.py /image_1_path /image_2_path --model_dir /path/to/best_model --checkpoint_dir /path/to checkpoint
```

The test_net file outputs different distance metrics and a visualization result is saved into the ./test directory.  


# Results and Improvements

Results are improvements are explained in the report.pdf

