import random
import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from central_reservoir.models import i3d
from central_reservoir.augmentations import preprocessing_volume

from absl import flags
from absl import app

path_to_pickle = '/home/rohitsaha/Human_video_frame_labels.pkl'
with open(path_to_pickle, 'rb') as handle:
    videoname_frame = pickle.load(handle)

# Get test shards from bucket
test_shards_path = 'gs://serrelab/behavior_core_mice/TF_records/Human/2019-03-08_tanner_nih_annotated/'
shards = tf.gfile.Glob(
    os.path.join(
        test_shards_path,
        'test*'))
print('{} testing shards found'.format(len(shards)))
test_examples = 18500
batch_size = 20
print('Testing examples: {}'.format(test_examples))

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_folder_name',
    default='tanner_nih_4.5k_on_v2_256_crbn_4096batch_1000epochs_1iters_adam_1e-5lr',
    help='To mention the model path')

flags.DEFINE_integer(
    'step',
    default=14000,
    help='To specify the checkpoint')


BEHAVIOR_INDICES = {
    0:"drink",
    1:"eat",
    2:"groom",
    3:"hang",
    4:"sniff",
    5:"rear",
    6:"rest",
    7:"walk",
    8:"eathand"}

behaviors = [
    'drink',
    'eat',
    'groom',
    'hang',
    'sniff',
    'rear',
    'rest',
    'walk',
    'eathand']

slack_window = 20
each_side = slack_window / 2

def read_pickle(fi):
    with open(fi, 'rb') as handle:
        a = pickle.load(handle)
    return a

def temporal(fnames, fnumbers, top_class_batch, labels_batch):
    temporal_top_class_batch = []
    for i in range(len(fnames)):
        all_frames = videoname_frame[fnames[i]]
        frame_in_focus = fnumbers[i]

        get_window = all_frames[
            frame_in_focus - each_side :\
            frame_in_focus + each_side + 1]

        behav_id_in_focus = top_class_batch[i]
        behav_in_focus = BEHAVIOR_INDICES[behav_id_in_focus]

        if behav_in_focus in get_window:
            temporal_top_class_batch.append(labels_batch[i])
        else:
            temporal_top_class_batch.append(top_class_batch[i])

    return temporal_top_class_batch


def plot_confusion_matrix(png_path='',
            cnf_matrix=None,
            classes=[],
            annot_1='Ground_truth',
            annot_2='Predictions',
            balanced_acc=0.0,
            title='',
            cmap=plt.cm.Blues):

    plt.figure()
    plt.imshow(
        cnf_matrix,
        interpolation='nearest',
        cmap=cmap)

    title = title + ', Balanced acc: {}'.format(round(balanced_acc))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(cnf_matrix))

    plt.xticks(
        tick_marks,
        classes,
        rotation=45)
    plt.yticks(
        tick_marks,
        classes)

    fmt = '.2f'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                horizontalalignment='center',
                color='white' if cnf_matrix[i, j] > thresh else 'black')

    plt.ylabel(annot_1)
    plt.xlabel(annot_2)
    plt.tight_layout()

    plt.savefig(png_path)

def get_bal_acc(matrix):
    norm_matrix = np.zeros(
        (
            len(matrix),
            len(matrix)),
        dtype=np.float32)

    for i in range(len(matrix)):
        get_li = matrix[i].astype('float')
        sum_li = sum(get_li)
        if sum_li != 0:
            norm_matrix[i] = get_li / sum_li
        else:
            norm_matrix[i] = [0.0 for i in range(len(matrix))]

    diagonal, good_classes = 0, 0
    for i in range(len(norm_matrix)):
        good_classes += 1
        diagonal += norm_matrix[i][i]

    bal_acc = (diagonal / float(good_classes)) * 100.0

    return bal_acc, norm_matrix

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, ser = reader.read(filename_queue)

    keys_to_features = {
        'data/chunk': tf.FixedLenFeature(
            [],
            tf.string),
        'data/label': tf.FixedLenFeature(
            [],
            tf.int64),
        'data/video_name': tf.FixedLenFeature(
            [],
            tf.string),
        'data/frame_number': tf.FixedLenFeature(
            [],
            tf.int64)}

    parsed = tf.parse_single_example(
        ser,
        features=keys_to_features)

    video = tf.decode_raw(
        parsed['data/chunk'],
        tf.uint8)

    label = tf.cast(
        parsed['data/label'],
        tf.int32)

    video_name = parsed['data/video_name']

    frame_number = tf.cast(
        parsed['data/frame_number'],
        tf.int32)

    height, width = 256, 256

    video = preprocessing_volume.preprocess_volume(
        volume=video,
        num_frames=16,
        height=height,
        width=width,
        is_training=False,
        target_image_size=224,
        use_bfloat16=False,
        list_of_augmentations=['random_crop'])

    videos, labels, video_names, frame_numbers = tf.train.batch([video, label, video_name, frame_number],
        batch_size=batch_size,
        capacity=30,
        num_threads=1)

    return videos, labels, video_names, frame_numbers

def main(unused_argv):

    all_preds, all_ground, all_temporal_preds = [], [], []

    common_path = 'gs://serrelab/behavior_core_mice/Model_runs/'
    ckpt_path = os.path.join(
        common_path,
        FLAGS.model_folder_name,
        'model.ckpt-{}'.format(FLAGS.step))
    meta_path = os.path.join(
        common_path,
        FLAGS.model_folder_name,
        'model.ckpt-{}.meta'.format(FLAGS.step))

    with tf.Session().as_default() as sess:
        filename_queue = tf.train.string_input_producer(
            shards,
            num_epochs=None)
        video, label, filename, frame_number = read_and_decode(filename_queue)
        label = tf.one_hot(label, 9, dtype=tf.float32)

        network = i3d.InceptionI3d(
            final_endpoint='Logits',
            use_batch_norm=True,
            use_cross_replica_batch_norm=True,
            num_classes=9,
            spatial_squeeze=True,
            dropout_keep_prob=0.7)

        logits, end_points = network(
            inputs=video,
            is_training=False)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        try:
            for i in range(int(test_examples/batch_size)):
                preds, labs, fname, fnumber = sess.run(
                    [logits, label, filename, frame_number])
                preds_max = list(np.argmax(preds, axis=-1))
                labs_max = list(np.argmax(labs, axis=-1))
                all_preds += preds_max
                all_ground += labs_max

                temporal_top_class_batch = temporal(
                    fname,
                    fnumber,
                    preds_max,
                    labs_max)

                all_temporal_preds += temporal_top_class_batch

                if  ( ((i+1)*batch_size) % 50 == 0):
                    print('{}/{} completed'.format((i+1)*batch_size, test_examples))

            coord.request_stop()

        except:
            print('{} examples covered'.format(i))

    print('Testing complete')

    ground_behav = [BEHAVIOR_INDICES[i] for i in all_ground]
    preds_behav = [BEHAVIOR_INDICES[i] for i in all_preds]
    temporal_preds_behav = [BEHAVIOR_INDICES[i] for i in all_temporal_preds]

    with open('gt.pkl', 'wb') as handle:
        pickle.dump(ground_behav, handle)
    with open('preds.pkl', 'wb') as handle:
        pickle.dump(preds_behav, handle)
    with open('temporal_preds.pkl', 'wb') as handle:
        pickle.dump(temporal_preds_behav, handle)

    cnf_matrix_ground_preds = confusion_matrix(ground_behav, preds_behav)
    cnf_matrix_ground_temporal_preds = confusion_matrix(ground_behav, temporal_preds_behav)
    np.set_printoptions(precision=2)
    gp_balacc, gp_normmatrix = get_bal_acc(cnf_matrix_ground_preds)
    gtp_balacc, gtp_normmatrix = get_bal_acc(cnf_matrix_ground_temporal_preds)

    plot_confusion_matrix(png_path='/home/rohitsaha/tanner_ground_preds.png',
        cnf_matrix=gp_normmatrix,
        classes=behaviors,
        annot_1='Ground_truth',
        annot_2='Predictions',
        balanced_acc=gp_balacc,
        title='Tanner\'s held-out video stats: ',
        cmap=plt.cm.Blues)

    plot_confusion_matrix(png_path='/home/rohitsaha/tanner_ground_temporal_preds_slack20.png',
        cnf_matrix=gtp_normmatrix,
        classes=behaviors,
        annot_1='Ground_truth',
        annot_2='Predictions',
        balanced_acc=gtp_balacc,
        title='Tanner\'s held-out video stats, slack=20, : ',
        cmap=plt.cm.Blues)


if __name__ == '__main__':
    app.run(main)
