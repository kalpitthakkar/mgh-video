import yaml

import os
import tensorflow as tf

from annotate_gui import AnnotationGUI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    left_clip = tf.reshape(left_clip, [16, 256, 256, 3])
    right_clip = tf.reshape(right_clip, [16, 256, 256, 3])
    left_clips, right_clips, labels, video_name, frame_number = tf.train.shuffle_batch(
        [left_clip, right_clip, label, video_name, frame_number],
        batch_size=1,
        capacity=100,
        min_after_dequeue=50,
        num_threads=2)

    return left_clips, right_clips, labels, video_name, frame_number

if __name__ == '__main__':
    with open('ksu_mice.yaml', 'r') as f:
        out = yaml.load(f)
    data_dir = out['data_dir']
    output_dir = out['output_dir']

    data_shards = [os.path.join(data_dir, fi) for fi in os.listdir(data_dir)]
    lclips, rclips, labs, vname, fnum = data_read(data_shards)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        ctr = 0
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(out['samples_to_annotate']):
            l, r, lab, vidname, frame = sess.run([lclips, rclips, labs, vname, fnum])
            # Create a gui for each sample separately => cannot save the data,
            # else it'll occupy too much space
            gui = AnnotationGUI(data=[l, r, lab, vidname, frame], output_dir=output_dir,
                    with_annots=False, annots_file_ext='csv',
                    yaml_config='ksu_mice.yaml')

            root = gui.build_gui()
            root.mainloop()
            
            # Can access the annotations using gui.left_annots and gui.right_annots.
            # They are dataframes with columns ['video_path', 'frame_n', 'attention_loc']
            # 'video_path' is just the video name and other columns are self-explanatory.
            # Use them to calculate your masks and write everything to new tfrecords.
            
            # <Your Masks calculation and TFRecords code>

            ctr += 1
        coord.request_stop()
        coord.join(threads)
        sess.close()
