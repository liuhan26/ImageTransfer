# -*- coding: utf-8 -*-
import os
import pprint
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from random import shuffle
import argparse
import cv2

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_float("weight_decay", 1e-5, "Weight decay for l2 loss")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("z_dim", 100, "Size of Noise embedding")
#flags.DEFINE_integer("class_embedding_size", 5, "Size of class embedding")
# flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 100, "The interval of saveing checkpoints. [200]")
flags.DEFINE_integer("imageEncoder_steps", 30000, "Number of train steps for image encoder")
flags.DEFINE_string("dataset", "lock3dface", "The name of dataset [celebA, obama_hillary, svhn_inpainting]")
flags.DEFINE_string("checkpoint_dir", "data/Models", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "data/samples", "Directory name to s ave the image samples [samples]")
# flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system('mkdir data')
os.system('mkdir {}'.format(FLAGS.sample_dir))
os.system('mkdir {}'.format(FLAGS.sample_dir+'/step1'))
os.system('mkdir {}'.format(FLAGS.sample_dir+'/step2'))


import rgbd.rgbd_loader as data_loader
# from model import *
import CoGAN_model as model
from utils import *

if FLAGS.image_size == 64:
    generator = model.generator
    discriminator = model.discriminator
    imageEncoder = model.imageEncoder

else:
    raise Exception("image_size should be 64 or 256")

def train_ac_gan():
    z_dim = FLAGS.z_dim
    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    real_images1 =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim], name='real_images1')
    real_images2 =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim], name='real_images2')

    # z --> generator for training
    net_g1, _ = generator(z_noise, is_train=True, share_params=False, reuse=False, name='G1')
    net_g2, _ = generator(z_noise, is_train=True, share_params=True, reuse=False, name='G2')
    
    # generated fake images --> discriminator
    net_d1, d_logits_fake1, _, d_logits_fake_class1, _ = discriminator(net_g1.outputs, is_train=True, share_params=False, reuse=False, name='D1')
    net_d2, d_logits_fake2, _, d_logits_fake_class2, _ = discriminator(net_g2.outputs, is_train=True, share_params=True, reuse=False, name='D2')
    # real images --> discriminator
    _, d_logits_real1, _, d_logits_real_class1, _ = discriminator(real_images1, is_train=True, share_params=True, reuse=True, name='D1')
    _, d_logits_real2, _, d_logits_real_class2, _ = discriminator(real_images2, is_train=True, share_params=True, reuse=True, name='D2')
    # sample_z --> generator for evaluation, set is_train to False
    net_sample_g1, _ = generator(z_noise, is_train=False, share_params=True, reuse=True, name='G1')
    net_sample_g2, _ = generator(z_noise, is_train=False, share_params=True, reuse=True, name='G2')


    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1
    d_loss_real1 = tl.cost.sigmoid_cross_entropy(d_logits_real1, tf.ones_like(d_logits_real1), name='dreal1')
    d_loss_real2 = tl.cost.sigmoid_cross_entropy(d_logits_real2, tf.ones_like(d_logits_real2), name='dreal2')
    # discriminator: images from generator (fake) are labelled as 0  
    d_loss_fake1 = tl.cost.sigmoid_cross_entropy(d_logits_fake1, tf.zeros_like(d_logits_fake1), name='dfake')
    d_loss1 = d_loss_real1 + d_loss_fake1
    d_loss_fake2 = tl.cost.sigmoid_cross_entropy(d_logits_fake2, tf.zeros_like(d_logits_fake2), name='dfake')
    d_loss2 = d_loss_real2 + d_loss_fake2
    # generator: try to make the the fake images look real (1)
    g_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake1, labels=tf.ones_like(d_logits_fake1)))
    g_loss1 = g_loss_fake1
    g_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake2, labels=tf.ones_like(d_logits_fake2)))
    g_loss2 = g_loss_fake2
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'g' in var.name]
    d_vars = [var for var in t_vars if 'd' in var.name]
    d_loss = d_loss1 + d_loss2
    g_loss = g_loss1 + g_loss2
    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=g_vars)

    sess=tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.5)
    sess.run(tf.initialize_all_variables())

    net_g1_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g1.npz'.format(FLAGS.dataset))
    net_d1_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_d1.npz'.format(FLAGS.dataset))
    net_g2_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g2.npz'.format(FLAGS.dataset))
    net_d2_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_d2.npz'.format(FLAGS.dataset))
    # net_e_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_e.npz'.format(FLAGS.dataset))

    if not FLAGS.retrain:
        if not (os.path.exists(net_g1_name) and os.path.exists(net_d1_name)):
            print("[!] Could not load weights from npz files")
        else:
            net_g1_loaded_params = tl.files.load_npz(name=net_g1_name)
            net_d1_loaded_params = tl.files.load_npz(name=net_d1_name)
            net_g2_loaded_params = tl.files.load_npz(name=net_g2_name)
            net_d2_loaded_params = tl.files.load_npz(name=net_d2_name)
            # net_e_loaded_params = tl.files.load_npz(name=net_e_name)
            tl.files.assign_params(sess, net_g1_loaded_params, net_g1)
            tl.files.assign_params(sess, net_d1_loaded_params, net_d1)
            tl.files.assign_params(sess, net_g2_loaded_params, net_g2)
            tl.files.assign_params(sess, net_d2_loaded_params, net_d2)
            print("[*] Loading checkpoints SUCCESS!")
    else:
        print("[*] Retraining AC GAN")

    class1_files, class2_files = data_loader.load_data(FLAGS.dataset)
    all_files = class1_files + class2_files
    total_batches = len(all_files)/FLAGS.batch_size
    shuffle(all_files)
    print("all_files", len(all_files))

    print("Total_batches", total_batches)

    for epoch in range(FLAGS.epoch):
        for bn in range(0, int(total_batches)):

            idex1 = get_random_int(min=0, max=len(class1_files)-1, number=int(FLAGS.batch_size))
            idex2 = get_random_int(min=0, max=len(class2_files)-1, number=int(FLAGS.batch_size))
            batch_files1 = [ class1_files[i] for i in idex1]
            batch_files2 = [ class2_files[i] for i in idex2]
            batch_images1 = threading_data(batch_files1, fn=get_image_fn)
            batch_images2 = threading_data(batch_files2, fn=get_image_fn)
            batch_images1 = threading_data(batch_images1, fn=distort_fn)
            batch_images2 = threading_data(batch_images2, fn=distort_fn)

            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                z_noise: batch_z,             
                real_images1: batch_images1,
                real_images2: batch_images2
            })

            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={
                    z_noise: batch_z,
                })

            print("d_loss={}\t g_loss={}\t epoch={}\t batch_no={}\t total_batches={}".format(errD, errG, epoch, bn, total_batches))

            if bn % FLAGS.save_step == 0:
                print("[*] Saving Models...")

                tl.files.save_npz(net_g1.all_params, name=net_g1_name, sess=sess)
                tl.files.save_npz(net_d1.all_params, name=net_d1_name, sess=sess)
                tl.files.save_npz(net_g2.all_params, name=net_g2_name, sess=sess)
                tl.files.save_npz(net_d2.all_params, name=net_d2_name, sess=sess)
                # Saving after each iteration
                tl.files.save_npz(net_g1.all_params, name=net_g1_name + "_" + str(epoch), sess=sess)
                tl.files.save_npz(net_d1.all_params, name=net_d1_name + "_" + str(epoch), sess=sess)               
                tl.files.save_npz(net_g2.all_params, name=net_g2_name + "_" + str(epoch), sess=sess)
                tl.files.save_npz(net_d2.all_params, name=net_d2_name + "_" + str(epoch), sess=sess)        
                print("[*] Models saved")

                generated_samples, generated_samples_other_class = sess.run([net_sample_g1.outputs, net_sample_g2.outputs], feed_dict={
                    z_noise: batch_z,
                })

                combine_and_save_image_sets( [batch_images1,batch_images2, generated_samples, generated_samples_other_class], FLAGS.sample_dir+'/step1')

def train_imageEncoder():
    z_dim = FLAGS.z_dim

    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')

    net_z_classes = InputLayer(inputs = tf.one_hot(z_classes, 2), name ='classes_embedding')


    net_g, _ = generator(tf.concat([z_noise, net_z_classes.outputs], 1), is_train=False, reuse=False)

    net_p = imageEncoder(net_g.outputs, is_train=True)

    net_g2, _ = generator(tf.concat([net_p.outputs, net_z_classes.outputs], 1), is_train=False, reuse=True)

    t_vars = tf.trainable_variables()
    p_vars = [var for var in t_vars if 'imageEncoder' in var.name]

    p_loss = tf.reduce_mean( tf.square( tf.subtract( net_p.outputs, z_noise) ))

    p_optim = tf.train.AdamOptimizer(FLAGS.learning_rate/2, beta1=FLAGS.beta1) \
                  .minimize(p_loss, var_list=p_vars)

    sess = tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.5)

    tl.layers.initialize_global_variables(sess)

    # RESTORE THE TRAINED AC_GAN

    net_g_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g.npz'.format(FLAGS.dataset))

    net_e_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_e.npz'.format(FLAGS.dataset))

    if not (os.path.exists(net_g_name) and os.path.exists(net_e_name)):
        print("[!] Loading checkpoints failed!")
        return
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)

        tl.files.assign_params(sess, net_g_loaded_params, net_g2)
        tl.files.assign_params(sess, net_e_loaded_params, net_z_classes)

        print("[*] Loading checkpoints SUCCESS!")

    net_p_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_p.npz'.format(FLAGS.dataset))
    if not FLAGS.retrain:
        net_p_loaded_params = tl.files.load_npz(name=net_p_name)
        tl.files.assign_params(sess, net_p_loaded_params, net_p)
        print("[*] Loaded Pretrained Image Encoder!")
    else:
        print("[*] Retraining ImageEncoder")


    model_no = 0
    for step in range(0, FLAGS.imageEncoder_steps):
        batch_z_classes = [0 if random.random() > 0.5 else 1 for i in range(FLAGS.batch_size)]
        batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

        batch_images, gen_images, _, errP = sess.run([net_g.outputs, net_g2.outputs, p_optim, p_loss], feed_dict={
            z_noise : batch_z,
            z_classes : batch_z_classes,
        })

        print("p_loss={}\t step_no={}\t total_steps={}".format(errP, step, FLAGS.imageEncoder_steps))

        if step % FLAGS.sample_step == 0:
            print("[*] Sampling images")
            combine_and_save_image_sets([batch_images, gen_images], FLAGS.sample_dir+'/step2')

        if step % 2000 == 0:
            model_no += 1

        if step % FLAGS.save_step == 0:
            print("[*] Saving Model")
            tl.files.save_npz(net_p.all_params, name=net_p_name, sess=sess)
            tl.files.save_npz(net_p.all_params, name=net_p_name + "_" + str(model_no), sess=sess)
            print("[*] Model p(encoder) saved")

def main(_):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_step', type=str, default="ac_gan",
                       help='Step of the training : ac_gan, imageEncoder')

    parser.add_argument('--retrain', type=int, default=0,
                       help='Set 0 for using pre-trained model, 1 for retraining the model')

    args = parser.parse_args()

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    FLAGS.retrain = args.retrain == 1

    if args.train_step == "ac_gan":
        train_ac_gan()

    elif args.train_step == "imageEncoder":
        train_imageEncoder()

if __name__ == '__main__':

    tf.app.run()
