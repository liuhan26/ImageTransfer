import pprint
import rgbd.rgbd_loader as data_loader
from CoGAN_model import *
from utils import *
from matplotlib import pyplot as plt
import scipy.misc as sci
pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 1, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("z_dim", 100, "Size of Noise embedding")
flags.DEFINE_integer("class_embedding_size", 5, "Size of class embedding")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "lock3dface", "The name of dataset [celebA, obama_hillary, svhn_inpainting]")
flags.DEFINE_string("checkpoint_dir", "data/Models", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "data/samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("test_step", 2, "1 for generate samples, 2 for translate images for recognition")
FLAGS = flags.FLAGS

os.system('mkdir {}'.format(FLAGS.sample_dir + '/step3'))


def main():
    real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim],
                                 name='real_images')
    net_p = imageEncoder(real_images, is_train=False)
    net_g, _ = generator(net_p.outputs, is_train=False, share_params=False, reuse=False, name='G2')

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    net_g_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g2.npz'.format(FLAGS.dataset))
    net_p_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_p.npz'.format(FLAGS.dataset))

    if not (os.path.exists(net_g_name) and os.path.exists(net_p_name)):
        print("[!] Loading checkpoints failed!")
        return
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_p_loaded_params = tl.files.load_npz(name=net_p_name)

        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_p_loaded_params, net_p)

        print("[*] Loading checkpoints SUCCESS!")

    if FLAGS.test_step == 1:
        class1_files, class2_files, class_flag = data_loader.load_data(FLAGS.dataset)

        batch_files = class1_files[0:FLAGS.batch_size]
        batch_images = threading_data(batch_files, fn=get_image_fn)
        batch_images = threading_data(batch_images, fn=distort_fn)

        gen_images = sess.run(net_g.outputs, feed_dict={
            real_images: batch_images
        })

        combine_and_save_image_sets([batch_images, gen_images], FLAGS.sample_dir + '/step3')
    else:
        class_files = data_loader.load_data(FLAGS.dataset)

        root = '/media/liuhan/xiangziBRL/Lock3DFace/TestImage/fake2_depth/'
        i = 0
        for path in class_files:
            # if not os.path.exists(root + path[6:12]):
            #     os.mkdir(root + path[6:12])
            batch_images = threading_data(["/media/liuhan/xiangziBRL/Lock3DFace/croppedData_LightenedCNN/"+path], fn=get_image_fn)
            batch_images = threading_data(batch_images, fn=distort_fn)
            gen_images = sess.run(net_g.outputs, feed_dict={
                    real_images: batch_images
            })
            img = sci.imresize(gen_images[0], (128, 128))
            str = path.split("COLOR")
            plt.imsave(root+str[0][13:]+"DEPTH"+str[1],img)
            i = i + 1
            print(i)

    print("[*] Translation Complete")


if __name__ == '__main__':
    main()
