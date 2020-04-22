import argparse
import os
import numpy as np
import tensorflow as tf
from data_utils import *
from train_ops import network, contrastive_loss, identity_loss, mean_average_precision, accuracy, accuracy_metric, pretrained_network
import time
# tfrecords_path='./tf_records_data/'
# BATCH_SIZE = 8

class SiameseNet(object):
    def __init__(self, sess, args):
        """
        :param sess:
        :param args:
        """
        self.args = args
        self.sess = sess
        self.batch_size = args.batch_size
        self.dataset_dir = args.dataset_dir
        self.tf_record_dir = args.tf_record_dir
        self.loss_fn = args.loss
        self.data_augment = args.data_augment

        if args.network_type == "pretrained":
            self.net = pretrained_network  
        else:
            self.net = network
            
        self.loss_fn_1 = identity_loss
        self.loss_fn_2 = contrastive_loss
        
        self.loss = args.loss
        self.network_type = args.network_type 
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_iter_in_epoch = 0
        self.validation_intv = args.validation_intv
        self.summary_intv = args.summary_intv
        _, model_dir = self.get_model_str()
        self.log_dir = os.path.join(args.log_dir, model_dir)
        self._build_data(args)
        self._build_model(args)
        self._build_optim(args)
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def _build_data(self, args):
        """Build the dataset structures for train/valid.
        """
        self.inputs = {}
        self.targets = {}
        self.iter_handle = {}
        self.dataset = {} 

        self.handle = tf.placeholder(tf.string, shape=[])
        if args.network_type == "pretrained":
            pretrained = True
        else:
            pretrained = False
        
        for _set_type in ["train", "valid"]:
            if _set_type == "train":
                filename = 'train.tfrecords'
                train = True
            else:
                filename = 'val.tfrecords'
                train = False
                
            self.dataset[_set_type] =  get_data(
                self.tf_record_dir + '/' + filename,
                self.batch_size,
                self.data_augment,
                pretrained,
                train
            )
            
            self.iter_handle[_set_type] = self.dataset[_set_type].make_one_shot_iterator()
            
            iterator = tf.data.Iterator.from_string_handle(
                self.handle, self.dataset["train"].output_types, self.dataset["train"].output_shapes)
            next_batch = iterator.get_next()
            self.inputs[_set_type], self.targets[_set_type] = next_batch

        data_filename = os.path.join(args.tf_record_dir, 'data_summary.txt')
        with tf.gfile.Open(data_filename, 'r') as f:
            self.num_validation = int(f.readline())
            num_dataset = int(f.readline())
        self.num_training = int(int(num_dataset) - int(self.num_validation))
        print(f'Number of Training Images {self.num_training}')
        print(f'Number of Validation Images {self.num_validation}')

        
    def _build_model(self, args):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # --------------------
        # Create different logits and class_probs depending on input
        self.logits = {}
        self.pred = {}
        self.identity_loss = {}
        self.contrastive_loss = {}
        self.total_loss = {}
        self.mean_avg_precision = {}
        self.mean_avg_precision_up = {}
        self.accuracy = {}
        self.accuracy_up = {}
        self.accuracy_metric = {}
        self.summary = {}
        self.writer = {}
        self.output = {}
        self.left_feat = {}
        self.right_feat = {}
  
        # --------------------
        # Create network with appropriate inputs
        for _set_type in ["train", "valid"]:
            # Alias for easy manipulation
            x, x_label, _  = self.inputs[_set_type]
            y, y_label, _ = self.targets[_set_type]

            print("X:", x.shape)
            print("y:", y.shape)
            
            summmary_list = []
            # Logits
            self.output[_set_type] = self.net(
                left_im=x,
                right_im=y,
                is_training=self.is_training,
                batch_size=self.batch_size
            )
            print ("[*] Net defiend") 
            self.logits[_set_type] = self.output[_set_type][0]
            self.left_feat[_set_type] = self.output[_set_type][1]
            self.right_feat[_set_type] = self.output[_set_type][2]

             # Loss
            self.contrastive_loss[_set_type] = self.loss_fn_2(
                left_feat=self.left_feat[_set_type], 
                right_feat=self.right_feat[_set_type], 
                y=self.logits[_set_type], 
                left_label=x_label, 
                right_label=y_label, 
                margin=1.0, 
                use_loss=True)
        
            self.identity_loss[_set_type] = self.loss_fn_1(
                logits=self.logits[_set_type], 
                left_label=x_label, 
                right_label=y_label
            )
            
            print (f"loss: {args.loss}")
            if args.loss == "combined":
                self.total_loss[_set_type] = self.contrastive_loss[_set_type] + self.identity_loss[_set_type]
            elif args.loss == "identity":
                self.total_loss[_set_type] = self.identity_loss[_set_type]
            elif args.loss == "contrastive":
                self.total_loss[_set_type] = self.contrastive_loss[_set_type]
                
            self.mean_avg_precision[_set_type], self.mean_avg_precision_up[_set_type] = mean_average_precision(
                 logits=self.logits[_set_type], 
                 left_label=x_label, 
                 right_label=y_label
            )
            
            
            self.accuracy_metric[_set_type] = accuracy_metric(
                 logits=self.logits[_set_type], 
                 left_label=x_label, 
                 right_label=y_label
            )
            self.accuracy[_set_type], self.accuracy_up[_set_type] = accuracy(
                 logits=self.logits[_set_type], 
                 left_label=x_label, 
                 right_label=y_label
            )
            # Summary
            # summmary_list = []
            summmary_list += [tf.summary.image(
                "input/{}".format(_set_type), x)]
            summmary_list += [tf.summary.image(
                "target/{}".format(_set_type),y)]
            summmary_list += [tf.summary.scalar(
                "contrastive_loss/{}".format(_set_type),
                self.contrastive_loss[_set_type])]
            summmary_list += [tf.summary.scalar(
                "identity_loss/{}".format(_set_type),
                self.identity_loss[_set_type])]
            summmary_list += [tf.summary.scalar(
                "total_loss/{}".format(_set_type),
                self.total_loss[_set_type])]
            summmary_list += [tf.summary.scalar(
                "mean_average_precision/{}".format(_set_type),
                self.mean_avg_precision[_set_type])]
            summmary_list += [tf.summary.scalar(
                "accuracy/{}".format(_set_type),
                self.accuracy[_set_type])]
            summmary_list += [tf.summary.scalar(
                "accuracy_metric/{}".format(_set_type),
                self.accuracy_metric[_set_type])]
            
            self.summary[_set_type] = tf.summary.merge(summmary_list)
            # Also build the writer for summary here
            self.writer[_set_type] = tf.summary.FileWriter(
                os.path.join(self.log_dir, _set_type), self.sess.graph)

        # Reset op for initializing local variables in metric
        self.reset_local_var_op = tf.local_variables_initializer()

        # Variable to store best validation
        self.best_acc = tf.Variable(
            initial_value=0, dtype=tf.float32,
            name="best_acc", trainable=False)
        self.new_acc = tf.placeholder(
            tf.float32, shape=(),
            name='new_acc')
        self.assign_best = tf.assign(self.best_acc, self.new_acc)
        
#         if args.network_type == "pretrained":
#             self.t_vars = tf.trainable_variables(scope="trainable_section")
#         else:
        self.t_vars = tf.trainable_variables()
            
        print("[*] Trainable variables:")
        for var in self.t_vars:
            print(var.name)
  
    def _build_optim(self, args):
        self.global_step = tf.Variable(initial_value=0, dtype=tf.int64,
                                       name="global_step")
        self.optim = tf.train.AdamOptimizer(
            args.lr, beta1=args.beta1
        ).minimize(
            self.total_loss["train"], var_list=self.t_vars,
            global_step=self.global_step
        )

    def train(self, args):
        """Train SiameseNet"""

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # counter = 0
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # We are using dataset API, no need to resume iteration
        # Reset dataset for training
        training_handle = self.sess.run(self.iter_handle["train"].string_handle())
        while True:

            # Fetch step
            step = self.sess.run(self.global_step)
            # Setup fetch dictionary
            fetch = {
                "optim": self.optim,
                "mean_avg_precision_up": self.mean_avg_precision_up["train"],
                "accuracy_up": self.accuracy_up["train"],
                "accuracy_metric": self.accuracy_metric["train"],
                "step": self.global_step
            }
            

            # Check summary intv
            # b_fetch_summary = (step == 0) or (
            #     ((step + 1) % self.summary_intv) == 0
            # )
            b_fetch_summary = ((step + 1) % self.summary_intv) == 0
            if b_fetch_summary:
                fetch["summary"] = self.summary["train"]

            # Run session
            res = self.sess.run(fetch, 
                                feed_dict={self.is_training: True, self.handle: training_handle}
                               )

            # Get summary every reporting interval
            if b_fetch_summary:
                self.writer["train"].add_summary(
                    summary=res["summary"],
                    global_step=res["step"]
                )
                # counter replaced by iterator
                epoch = (res["step"] * self.batch_size) // self.num_training
                
#                 cur_acc = self.sess.run(self.accuracy["train"])
                print(
                    "Epoch: {}, Iteration: {}, Accuracy: {}, Time: {}".format(
                        epoch, res["step"], res["accuracy_metric"], time.time() - start_time
                    )
                )
                
                # save model
                self.save(args.checkpoint_dir, res["step"])

            # The validation loop
            # b_validation = (step == 0) or (
            #     ((step + 1) % self.validation_intv) == 0
            # )
            b_validation = ((step + 1) % self.validation_intv) == 0
            # Perform validation
            if b_validation:
                # Compare current result with best validation
                best_acc = self.sess.run(
                    self.best_acc, feed_dict={self.is_training: False})
                # If first iteration, i.e. best_acc is zero, save
                # immediately
                if best_acc < 1e-5:
                    print("Savining model immediately for debug")
                    self.save(args.checkpoint_dir, res["step"], best=True)

                # Test on the entire dataset
                st_time = time.time()
                val_res = self.test_on_dataset(mode="valid", args=args)
                acc = val_res["accuracy"]
                print("time on validation:", time.time() - st_time)

                print("Validation: best {}, cur {}".format(
                    best_acc, acc))

                if acc > best_acc:
                    self.sess.run(
                        self.assign_best,
                        feed_dict={
                            self.new_acc: acc
                        }
                    )
                    print("Updated to new best {}".format(acc))
                    self.save(args.checkpoint_dir, res["step"], best=True)

            # Quit if we ran enough iterations
            if res["step"] >= args.max_iter:
                print('Reached maximum number of SGD iterations'
                      ' for this run ({:d})'.format(
                          args.max_iter))
                break

    def get_model_str(self, best=False):
        if best is True:
            model_name = "mars.best.model"
            model_dir = f"MARS_PERSON_REID_BEST_MODELS_{self.loss}_{self.network_type}_{self.batch_size}_{self.lr}_augment_{self.data_augment}" 
#             % (
#                 self.loss, self.network_type, self.batch_size, self.lr)
        else:
            model_name = "mars.model"
            model_dir = f"MARS_PERSON_REID_{self.loss}_{self.network_type}_{self.batch_size}_{self.lr}_augment_{self.data_augment}"
#             "%s_%s_%s" % (
#                 self.loss, self.network_type, self.batch_size, self.lr)
        print ("[*]", model_name, model_dir)

        return model_name, model_dir

    def save(self, checkpoint_dir, step, best=False):
        model_name, model_dir = self.get_model_str(best=best)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        # print ("SAVE--checkpoint_dir:", checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if best is True:
            self.best_saver.save(self.sess,
                                 os.path.join(checkpoint_dir, model_name),
                                 write_meta_graph=False,
                                 global_step=step)
        else:
            self.saver.save(self.sess,
                            os.path.join(checkpoint_dir, model_name),
                            write_meta_graph=False,
                            global_step=step)

    def load(self, checkpoint_dir, best=False):
        print(" [*] Reading checkpoint...")

        model_name, model_dir = self.get_model_str(best=best)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print("LOAD--checkpoint_dir:", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("LOAD--tf.train.get_checkpoint_state(checkpoint_dir):", ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            print("LOAD--ckpt.model_checkpoint_path:",
                  ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("LOAD--ckpt_name:", ckpt_name)
            if best is True:
                self.best_saver.restore(
                    self.sess, os.path.join(checkpoint_dir, ckpt_name))
            else:
                self.saver.restore(self.sess, os.path.join(
                    checkpoint_dir, ckpt_name))

            return True
        else:
            return False

    def test_on_dataset(self, mode, args):
        """ Internal test routine that returns the mean_iou """

        print ("[*] Validation -- ")
#         print(
#             "DEBUGME: Breakpoint here and check that everything "
#             "is identical, including args.")

        # We simply want to run on the whole data. Visualization can simply go
        # to the pipeline example, or another single frame run!

        # Reset local variables
        self.sess.run(self.reset_local_var_op)
        # Reset the dataset iterator
#         self.sess.run(self.reset_data[mode])
        validation_handle = self.sess.run(self.iter_handle[mode].string_handle())

        num_batch = int(np.ceil(
            self.num_validation / args.batch_size))

        fetch = {}
        precision = []
        accuracy_list = []
        for _i in range(num_batch):
            # Cumulate the mean_avg_precision
            fetch["mAP_update"] = self.mean_avg_precision_up[mode]
            fetch["accuracy_update"] = self.accuracy_up[mode]
            fetch["accuracy_metric"] = self.accuracy_metric[mode]
            fetch["step"] = self.global_step
            # Fetch also the summary for the first batch
            if _i == 0:
                fetch["summary"] = self.summary[mode]
            # Run
            res = self.sess.run(
                fetch, feed_dict={self.is_training: False, self.handle: validation_handle})
            # Save the first batch summary to report
            if "summary" in res:
                summary = res["summary"]
            accuracy_list.append(res["accuracy_metric"])
            
        mean_acc_metric = np.mean(np.array(accuracy_list))
                
        # Fetch the final mean AP, Accuracy
        mean_AP = self.sess.run(self.mean_avg_precision[mode])
        mean_acc = self.sess.run(self.accuracy[mode])
        # add result to create summary artificially and add IoU
        summary_new = tf.Summary(value=[
            tf.Summary.Value(
                tag="mean_average_precision/{}".format(mode),
                simple_value=mean_AP),
            tf.Summary.Value(
                tag="accuracy_metric/{}".format(mode),
                simple_value=mean_acc_metric),
             tf.Summary.Value(
                tag="accuracy/{}".format(mode),
                simple_value=mean_acc)
        ])
        self.writer[mode].add_summary(summary_new, res["step"])
        self.writer[mode].add_summary(summary, res["step"])
        self.writer[mode].flush()

        # --------------------
        # One final measure for time
        if args.report_time:
            # Reset the dataset iterator
#             self.sess.run(self.reset_data[mode])
            validation_handle = self.sess.run(self.iter_handle[mode].string_handle())
            # For all batches
            cum_time = 0
            for _i in range(num_batch):
                st_time = time.time()
                res = self.sess.run({"pred": self.pred[mode]},
                                    feed_dict={self.is_training: False, self.handle: validation_handle})
                ed_time = time.time()
                cum_time += ed_time - st_time
            print("avg time to test a single batch:", cum_time / num_batch)

        return {"mean_avg_precision": mean_AP, "accuracy": mean_acc}
    

    
parser = argparse.ArgumentParser(description='')

parser.add_argument('--max_iter', dest='max_iter', type=int, default=500000, help='# of max sgd iter')
parser.add_argument('--validation_intv', dest='validation_intv', type=int, default=1000, help='# of max sgd iter')
parser.add_argument('--summary_intv', dest='summary_intv', type=int, default=100, help='# of max sgd iter')
parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=10000000, help='# of max epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_iter_freq', dest='save_iter_freq', type=int, default=50, help='save a model every save_iter_freq sgd iter (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--subset_data', dest='subset_data', type=bool, default=False, help='if subset_data is true use 1000 sample to create the data: 1: true, 0: false')
parser.add_argument('--data_augment', dest='data_augment', type=bool, default=False, help='if data_augment is true use data augmentation: 1: true, 0: false')

parser.add_argument('--network_type', dest='network_type', default='siamese',help='pretrained, siamese')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./validation',help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log', help='test sample are saved here')
parser.add_argument('--loss', dest='loss', default='contrastive', help='contrastive, identity, combined')
# Dataset related
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data', help='path of the dataset')
parser.add_argument('--tf_record_dir', dest='tf_record_dir', default='./tf_record_dir', help='path of the dataset')
# parser.add_argument('--num_thread', dest='num_thread', type=int, default=16, help='# of threads')

# Report time
parser.add_argument('--report_time', dest='report_time', type=bool, default=False, help='Reports time taken on forward passes: 1: true, 0: false')

args = parser.parse_args()


# data_path = args.data
os.system(f'python3 data_tfrecord.py --tf_record_dir={args.tf_record_dir} --dataset_dir={args.dataset_dir} --subset_data={args.subset_data}')


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
#     if not os.path.exists(args.test_dir):
#         os.makedirs(args.test_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = SiameseNet(sess, args)
        model.train(args)

if __name__ == '__main__':
    tf.app.run()
