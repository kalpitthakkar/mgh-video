import yaml

import os
import pickle
import tensorflow as tf

import numpy as np
import create_mask_shards as cms
from annotate_gui import AnnotationGUI

def data_read(f):
    queue = tf.train.string_input_producer(f, num_epochs=1)
    reader = tf.TFRecordReader()
    _, ser = reader.read(queue)

    feature = {
        'data/label': tf.FixedLenFeature([], tf.int64),
        'data/left_clip': tf.FixedLenFeature([], tf.string),
        'data/right_clip': tf.FixedLenFeature([], tf.string),
        'data/video_name': tf.FixedLenFeature([], tf.string),
        'data/frame_number': tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_single_example(ser, features=feature)
    left_clip = tf.decode_raw(parsed['data/left_clip'], tf.uint8)
    right_clip = tf.decode_raw(parsed['data/right_clip'], tf.uint8)
    label = tf.cast(parsed['data/label'], tf.int32)
    video_name = parsed['data/video_name']
    frame_number = tf.cast(parsed['data/frame_number'], tf.int32)

    left_clip = tf.reshape(left_clip, [16, 512, 512, 3])
    right_clip = tf.reshape(right_clip, [16, 512, 512, 3])
    left_clips, right_clips, labels, video_name, frame_number = tf.train.shuffle_batch(
        [left_clip, right_clip, label, video_name, frame_number],
        batch_size=1,
        capacity=100,
        min_after_dequeue=50,
        num_threads=2)

    return left_clips, right_clips, labels, video_name, frame_number

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-((x-x0)**2 + (y-y0)**2) / fwhm**2)

def get_gaussian_mask(video, annot_framework):
    # annot_framework: Dataframe having the annotations
    T, H, W, C = video.shape
    attention_mask = np.zeros((T, H, W, 1), dtype=np.float32)
    for index, row in annot_framework.iterrows():
        size = video.shape[1]
        center = np.int32(row['attention_loc'].split('-'))
        mask = makeGaussian(size, fwhm=(size//8), center=center)
        attention_mask[index, ...] = np.expand_dims(mask, axis=-1)

    return attention_mask

if __name__ == '__main__':
    with open('ksu_mice.yaml', 'r') as f:
        out = yaml.load(f)
    video_dir = '/media/data_cifs/KSU_mice/videos/'
    videos = os.listdir(video_dir)
    data_dir = out['data_dir']
    output_dir = out['output_dir']
    pickle_dir = out['pickle_dir']
    tfr_output_dir = out['tfrecords_output_dir']

    if not os.path.exists(tfr_output_dir):
        os.makedirs(tfr_output_dir)
    # Initialize ctr with number of shards present in output dir
    ctr = len(os.listdir(tfr_output_dir)) + 1
    if not os.path.exists(pickle_dir):
        os.makedirs(os.path.split(pickle_dir)[0])
        visited = {}
    else:
        with open(pickle_dir, 'rb') as handle:
            visited = pickle.load(handle)

    data_shards = [os.path.join(data_dir, fi) for fi in os.listdir(data_dir)]
    name_shards = os.listdir(data_dir)
    lclips, rclips, labs, vname, fnum = data_read(data_shards)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        dir_name = os.path.basename(data_dir)
        if not os.path.exists(os.path.join(tfr_output_dir, dir_name)):
            os.makedirs(os.path.join(tfr_output_dir, dir_name))
        output_file = os.path.join(tfr_output_dir, dir_name, "train-{}".format(str(ctr)))
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.python_io.TFRecordWriter(output_file) 
        n_examples = 0
        for i in range(out['samples_to_annotate']):
            l, r, lab, vidname, frame = sess.run([lclips, rclips, labs, vname, fnum])
            l, r = l.squeeze(), r.squeeze()
            if vidname[0] not in videos:
                continue
            if vidname[0] in visited:
                if frame[0] in visited[vidname[0]]:
                    continue
            # Create a gui for each sample separately => cannot save the data,
            # else it'll occupy too much space
            gui = AnnotationGUI(data=[l, r, lab, vidname, frame], output_dir=output_dir,
                    with_annots=False, annots_file_ext='csv',
                    yaml_config='ksu_mice.yaml')

            root = gui.build_gui()
            root.mainloop()

            if vidname[0] not in visited:
                visited[vidname[0]] = []
            visited[vidname[0]].append(frame[0])

            # Can access the annotations using gui.left_annots and gui.right_annots.
            # They are dataframes with columns ['video_path', 'frame_n', 'attention_loc']
            # 'video_path' is just the video name and other columns are self-explanatory.
            # Use them to calculate your masks and write everything to new tfrecords.
            
            # <Your Masks calculation and TFRecords code>
            left_attention_mask = get_gaussian_mask(l, gui.left_annots)
            right_attention_mask = get_gaussian_mask(r, gui.right_annots)
            
            example = cms.feat_example(
                label=lab[0],
                left_clip=l,
                right_clip=r,
                left_mask=left_attention_mask,
                right_mask=right_attention_mask,
                filename=vidname[0],
                frame=frame[0]
            ) 

            n_examples += 1
            writer.write(example.SerializeToString())
            if n_examples % 1 == 0:
                # Code to write this to TFRecords
                # Close writer and create a new writer
                n_examples = 0
                writer.close()
                print("TFRecord file " + output_file + " written with 16 clips")
                with open(pickle_dir, 'wb') as handle:
                    pickle.dump(visited, handle)

                ctr += 1
                output_file = os.path.join(tfr_output_dir, dir_name, "train-{}".format(str(ctr)))
                # Open a new writer
                writer = tf.python_io.TFRecordWriter(output_file)
            
        coord.request_stop()
        coord.join(threads)
        sess.close()
