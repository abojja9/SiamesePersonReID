import PIL
from train_ops import network
import argparse
import os
import numpy as np
import tensorflow as tf
import time
import textwrap as tw
import matplotlib.pyplot as plt
from pathlib import Path

class EvalSiameseNet(object):
    def __init__(self, sess, args):
        """
        :param sess:
        :param args:
        """
        self.args = args
        self.sess = sess
      
        self.net = network
    
        self._build_model_data(args)
        self.saver = tf.train.Saver()

    def _build_model_data(self, args):
        """Build the dataset structures for train/valid.
        """
        
        img1 = PIL.Image.open(args.image_1_path)
        img2 = PIL.Image.open(args.image_2_path)
        
        self.img1 = np.array(img1)[np.newaxis, :, :, :]
        self.img2 = np.array(img2)[np.newaxis, :, :, :]
        
        # Normalize data
        self.img1 = self.img1/255.0
        self.img2 = self.img2/255.0
        
        
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
            dropout_rate=0,
            is_training=False,
            batch_size=args.test_batch_size
        )
    
    def contrastive_loss(self, left_feat, right_feat, logits):
        lft = np.array(left_feat[0])
        rgt = np.array(right_feat[0])
        l = lft - rgt
        distance = np.sqrt(np.sum((l) ** 2))
        similarity = logits[0] * np.square(distance)  
        dissimilarity = (1 - np.array(logits[0])) * np.square(np.max((1.0 - distance),0)) 
        contrast_loss = np.mean(dissimilarity + similarity)
        return {
            "distance": distance, 
            "similarity": similarity, 
            "dissimilarity": dissimilarity, 
            "contrast_loss": contrast_loss,
            "logits": logits
        }

        
    def test(self, args):
        """Test SiameseNet"""

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())     

        # counter = 0
        start_time = time.time()

        
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
        
        eval_time = time.time() - start_time
    
        
        y = res["logits"]
        left_feat = np.array(res["left_feat"][0])
        right_feat = np.array(res["right_feat"][0])
        
        metrics = self.contrastive_loss(res["left_feat"], res["right_feat"], y)
        print (f"Eval time: {eval_time}, Metrics: {metrics}")
        
            
        f_name = Path(args.image_1_path).name + Path(args.image_2_path).name
        
        show(self.img1[0], self.img2[0], metrics, args.test_dir+'/'+f_name+"_result.jpg")
       
    
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
        

def plt_fig_text(plt, facecolor, textstr):
    props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
    fig_txt = tw.fill(tw.dedent(textstr), width=80)
    plt.figtext(0.51, 0.05, fig_txt, horizontalalignment='center',
                fontsize=12, multialignment='center',
                bbox=dict(boxstyle="round", facecolor=facecolor,
                            ec="0.5", pad=0.5, alpha=1), fontweight='bold')
    return plt

        
def show(image_1, image_2, metrics, save_dir):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title(('Similarity Dist: %.3f,    Dissimilarity Dist: %.3f,    Euclidean Dist: %.3f \n'  % (metrics["similarity"], metrics["dissimilarity"], metrics["distance"])), loc='center')
    epsilon=1*10**(-8)

    x = 10000*metrics["similarity"]
    y = 10000*metrics["dissimilarity"] 

    if x < y:
        print ("[*] Images are Similar")
        textstr = 'Similar'
        plt = plt_fig_text(plt, "green", textstr)
    else:
        print ("[!] Images are Dissimilar...")
        textstr = 'Dissimilar'
        plt = plt_fig_text(plt, "red", textstr)

    plt.axis('off')
    ax1 = fig.add_subplot(1, 2, 1)
    l_im = np.array(image_1)
    ax1.imshow(l_im)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    r_im = np.array(image_2)
    ax2.imshow(r_im)
    ax2.axis('off')
    plt.savefig(save_dir)
    plt.show()


    
    

    
parser = argparse.ArgumentParser(description='')
parser.add_argument('image_1_path', type=str, help='path to the first image (left_im)')
parser.add_argument('image_2_path', type=str, help='Path to the second image (right_im)')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

parser.add_argument('--model_dir', dest='model_dir', default='',help='best model')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=1, help='# images in batch')
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
