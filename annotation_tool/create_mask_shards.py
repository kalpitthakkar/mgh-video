import tensorflow as tf
import numpy as np

def _int64_feature(value):
    '''To make tf.int64 datatype
    Args:
        value: 'Integer' to specify class id
    Returns:
        TFInt64 tensor
    '''
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[value]))


def _bytes_feature(value):
    '''To make tf.bytes datatype
    Args:
        value: 'Numpy' video clip of dtype uint8
    Returns:
        TFString tensor
    '''
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def feat_example(label=[], left_clip=[], right_clip=[],
                left_mask=[], right_mask=[],
                filename='', frame=0, transition=0):
    '''
    Define and return feature and example variables
        for TFR
    Args:
        label: 'Integer' or 'List' to mention class id
            of the clip
        left_clip: 'Numpy' video clip of dtype: uint8
        right_clip: 'Numpy' video clip of dtype: uint8
        left_mask: 'Numpy' video clip of annotated gaussian mask
            of dtype: uint8
        right_mask: 'Numpy' video clip of annotated gaussian mask
            of dtype uint8
        filename: 'String' to specify name of the video
        frame: 'Integer' to specify the frame number
        transition: 'Integer' to specify if transition
            data is present in experiment
    Returns:
        Protobuf example
    '''

    # Create a feature
    feature = {
        'data/label':_bytes_feature(
            tf.compat.as_bytes(
                label.tostring()))
            if transition else _int64_feature(
                label),
        'data/left_clip':_bytes_feature(
            tf.compat.as_bytes(
                left_clip.tostring())),
        'data/left_attention_mask': _bytes_feature(
            tf.compat.as_bytes(
                left_mask.tostring())),
        'data/right_attention_mask': _bytes_feature(
            tf.compat.as_bytes(
                right_mask.tostring())),
        'data/right_clip':_bytes_feature(
            tf.compat.as_bytes(
                right_clip.tostring())),
        'data/video_name':_bytes_feature(
            filename),
        'data/frame_number':_int64_feature(
            frame)}

    # Create an example protocol buffer
    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature))

    return example


