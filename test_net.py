import PIL
from train_ops import network

class EvalSiameseNet(object):
    def __init__(self, sess, args):
        """
        :param sess:
        :param args:
        """
        self.args = args
        self.sess = sess

        if args.network_type == "pretrained":
            self.net = pretrained  
        else:
            self.net = network
            
        self.loss_fn_1 = identity_loss
        self.loss_fn_2 = contrastive_loss
    
        self._build_data(args)
        self._build_model(args)
        self.saver = tf.train.Saver()

    def _build_model_data(self, args):
        """Build the dataset structures for train/valid.
        """
        
        img1 = PIL.Image.open(args.image_1_path)
        img2 = PIL.Image.open(args.image_2_path)
        
        self.img1 = np.array(img1)[np.newaxis, :, :, :]
        self.img2 = np.array(img2)[np.newaxis, :, :, :]
        
        
        self.x = tf.placeholder(tf.float32, [None, 256, 128, 3], 'left_im')
        self.y = tf.placeholder(tf.float32, [None, 256, 128, 3], 'right_im')
        self.x_label = tf.placeholder(tf.float32, [None, ], 'left_label')
        self.y_label = tf.placeholder(tf.float32, [None, ], 'right_label')
        self.global_step = tf.Variable(initial_value=0, dtype=tf.int64,
                                       name="global_step")

        print(np.shape(self.x), np.shape(self.y))
        self.logits, self.left_feat, self.right_feat = self.net(
                left_im=self.x,
                right_im=self.y,
                is_training=False,
                batch_size=args.test_batch_size)
        
                     # Loss
#         self.contrastive_loss = self.loss_fn_2(
#             left_feat=self.left_feat, 
#             right_feat=self.right_feat, 
#             y=self.logits, 
#             left_label=x_label, 
#             right_label=y_label, 
#             margin=0.2, 
#             use_loss=True)

#         self.identity_loss = self.loss_fn_1(
#             logits=self.logits, 
#             left_label=x_label, 
#             right_label=y_label
#         )
        
#     def _build_model(self, args):
#         self.is_training = tf.placeholder(tf.bool, name='is_training')

#         # --------------------
#         # Create different logits and class_probs depending on input
#         self.logits = {}
#         self.pred = {}
#         self.identity_loss = {}
#         self.contrastive_loss = {}
#         self.total_loss = {}
#         self.mean_avg_precision = {}
#         self.mean_avg_precision_up = {}
#         self.summary = {}
#         self.writer = {}
#         self.output = {}
#         self.left_feat = {}
#         self.right_feat = {}

#         # --------------------
#         # Create network with appropriate inputs
#         for _set_type in ["train", "valid"]:
#             # Alias for easy manipulation
#             x, x_label, _  = self.inputs[_set_type]
#             y, y_label, _ = self.targets[_set_type]

#             print("X:", x.shape)
#             print("y:", y.shape)

#             summmary_list = []
#             # Logits
#             self.output[_set_type] = self.net(
#                 left_im=x,
#                 right_im=y,
#                 is_training=self.is_training,
#                 batch_size=self.batch_size
#             )
#             print ("[*] Net defiend") 
#             self.logits[_set_type] = self.output[_set_type][0]
#             self.left_feat[_set_type] = self.output[_set_type][1]
#             self.right_feat[_set_type] = self.output[_set_type][2]

#              # Loss
#             self.contrastive_loss[_set_type] = self.loss_fn_2(
#                 left_feat=self.left_feat[_set_type], 
#                 right_feat=self.right_feat[_set_type], 
#                 y=self.logits[_set_type], 
#                 left_label=x_label, 
#                 right_label=y_label, 
#                 margin=0.2, 
#                 use_loss=True)
        
#             self.identity_loss[_set_type] = self.loss_fn_1(
#                 logits=self.logits[_set_type], 
#                 left_label=x_label, 
#                 right_label=y_label
#             )
            
#             print (f"loss: {args.loss}")
#             if args.loss == "combined":
#                 self.total_loss[_set_type] = self.contrastive_loss[_set_type] + self.identity_loss[_set_type]
#             elif args.loss == "identity":
#                 self.total_loss[_set_type] = self.identity_loss[_set_type]
#             elif args.loss == "contrastive":
#                 self.total_loss[_set_type] = self.contrastive_loss[_set_type]
                
#             print (f'[*] x_label: {x_label}, y_label {y_label}')
# #             self.mean_iou[_set_type], self.mean_iou_up[_set_type]
#             self.mean_avg_precision[_set_type], self.mean_avg_precision_up[_set_type] = mean_average_precision(
#                  logits=self.logits[_set_type], 
#                  left_label=x_label, 
#                  right_label=y_label
#             )
                
             
#             # Summary
#             # summmary_list = []
#             summmary_list += [tf.summary.image(
#                 "input/{}".format(_set_type), x)]
#             summmary_list += [tf.summary.image(
#                 "target/{}".format(_set_type),y)]
#             summmary_list += [tf.summary.scalar(
#                 "contrastive_loss/{}".format(_set_type),
#                 self.contrastive_loss[_set_type])]
#             summmary_list += [tf.summary.scalar(
#                 "identity_loss/{}".format(_set_type),
#                 self.identity_loss[_set_type])]
#             summmary_list += [tf.summary.scalar(
#                 "total_loss/{}".format(_set_type),
#                 self.total_loss[_set_type])]
#             summmary_list += [tf.summary.scalar(
#                 "mean_average_precision/{}".format(_set_type),
#                 self.mean_avg_precision[_set_type])]
            
#             self.summary[_set_type] = tf.summary.merge(summmary_list)
#             # Also build the writer for summary here
#             self.writer[_set_type] = tf.summary.FileWriter(
#                 os.path.join(args.log_dir, _set_type))

#         # Reset op for initializing local variables in metric
#         self.reset_local_var_op = tf.local_variables_initializer()

#         # Variable to store best validation
#         self.best_mAP = tf.Variable(
#             initial_value=0, dtype=tf.float32,
#             name="best_mAP", trainable=False)
#         self.new_mAP = tf.placeholder(
#             tf.float32, shape=(),
#             name='new_mAP')
#         self.assign_best = tf.assign(self.best_mAP, self.new_mAP)

#         self.t_vars = tf.trainable_variables()
#         for var in self.t_vars:
#             print(var.name)
  
    

    def test(self, args):
        """Test SiameseNet"""

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        

        # counter = 0
        start_time = time.time()
        
#         img = Image.open(img_one_path)
#         img = np.array(img)[np.newaxis, :, :, :]

#         img2 = Image.open(img_two_path)
#         img2 = np.array(img2)[np.newaxis,:,:,:]
        
        
        model_checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_dir)
        if self.load(model_checkpoint_dir):
            print(" [*] Model Loaded SUCCESSFULLY")
        else:
            print(" [!] Model Loading Failed...")
            
        # Setup fetch dictionary
        fetch = {
            "logits": self.logits,
            "left_feat": self.left_feat,
            "right_feat": self.right_feat
        }
        
        res = self.sess.run(fetch, 
                            feed_dict={self.x: self.img1, self.y: self.img2}
        )
        
        
        y = res["logits"]
        left_feat = np.array(res["left_feat"][0])
        right_feat = np.array(res["right_feat"][0])
        
        # Distance between the embedding vectors
        distance = tf.sqrt(tf.reduce_sum(tf.pow(left_feat - right_feat, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((0.5 - distance),
                                                       0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = tf.reduce_mean(dissimilarity + similarity) / 2
        
        
        diff = left_feat - right_feat
        distance = np.sqrt(np.sum((diff) ** 2))
        
        
        print(my_logits)
        print(np.shape(model_lf))
        print(np.shape(model_rg))

        lft = np.array(model_lf[0])
        rgt = np.array(model_rg[0])
        l = lft - rgt

        distance = np.sqrt(np.sum((l) ** 2))
        similarity = my_logits * np.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - np.array(my_logits[0])) * np.square(np.max((0.5 - distance),
                                                                        0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = np.mean(dissimilarity + similarity) / 2
        print('distance : ', distance)
        print('similarity : ', similarity)
        print('dissimilarity : ', dissimilarity)
        print('similarity_loss : ', similarity_loss)

                
        my_logits, model_lf, model_rg = sess.run([logits, model_left, model_right], \
                                                 feed_dict={left_input_im: img, right_input_im: img2})

        my_logits, model_lf, model_rg = sess.run([logits, model_left, model_right], \
                                                 feed_dict={left_input_im: img1, right_input_im: img2}

       

      
        while True:

            # Fetch step
            step = self.sess.run(self.global_step)
            
#             training_handle = self.sess.run(train_iterator)
#             validation_handle = self.sess.run(val_iterator.string_handle())
            

            # Setup fetch dictionary
            fetch = {
                "optim": self.optim,
                "mean_avg_precision": self.mean_avg_precision,
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
                print(
                    "Epoch: {}, Iteration: {}, Time: {}".format(
                        epoch, res["step"], time.time() - start_time
                    )
                )
                # save model
                self.save(args.checkpoint_dir, res["step"])
                #from IPython import embed; embed()

            # The validation loop
            # b_validation = (step == 0) or (
            #     ((step + 1) % self.validation_intv) == 0
            # )
            b_validation = ((step + 1) % self.validation_intv) == 0
            # Perform validation
            if b_validation:
                # Compare current result with best validation
                best_mAP = self.sess.run(
                    self.best_mAP, feed_dict={self.is_training: False})
                # If first iteration, i.e. best_mAP is zero, save
                # immediately
                if best_mAP < 1e-5:
                    print("Savining model immediately for debug")
                    self.save(args.checkpoint_dir, res["step"], best=True)

                # Test on the entire dataset
                st_time = time.time()
                val_res = self.test_on_dataset(mode="valid", args=args)
                mAP = val_res["mean_avg_precision"]
                print("time on validation:", time.time() - st_time)

                print("Validation: best {}, cur {}".format(
                    best_mAP, mAP))

                if mAP > best_mAP:
                    self.sess.run(
                        self.assign_best,
                        feed_dict={
                            self.new_mAP: mAP
                        }
                    )
                    print("Updated to new best {}".format(mAP))
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

        print("LOAD--checkpoint_dir:", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("LOAD--tf.train.get_checkpoint_state(checkpoint_dir):", ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            print("LOAD--ckpt.model_checkpoint_path:",
                  ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("LOAD--ckpt_name:", ckpt_name)
            self.saver.restore(self.sess, os.path.join(
                    checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test_on_dataset(self, mode, args):
        """ Internal test routine that returns the mean_iou """

        print ("[*] Validation -- ")
        print(
            "DEBUGME: Breakpoint here and check that everything "
            "is identical, including args.")

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
        for _i in range(num_batch):
            # Cumulate the mean_avg_precision
            fetch["update"] = self.mean_avg_precision_up[mode]
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
                
#             precision.append(res["update"])
           
#         mean_precision = np.mean(np.array(precision))
        # Fetch the final mean IOU
        mean_AP = self.sess.run(self.mean_avg_precision[mode])
        # add result to create summary artificially and add IoU
        summary_new = tf.Summary(value=[
            tf.Summary.Value(
                tag="mean_average_precision/{}".format(mode),
                simple_value=mean_AP)
#             tf.Summary.Value(
#                 tag="mean_avg_precision_list/{}".format(mode),
#                 simple_value=mean_precision),
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

        return {"mean_avg_precision": mean_AP}
    

    
parser = argparse.ArgumentParser(description='')
parser.add_argument('image_1_path', type=str, help='path to the first image (left_im)')
parser.add_argument('image_2_path', type=str, help='Path to the second image (right_im)')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in train batch size')
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=1, help='# images in test batch size')
parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
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

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = EvalSiameseNet(sess, args)
        model.test(args)

if __name__ == '__main__':
    tf.app.run()
