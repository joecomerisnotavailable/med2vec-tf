"""Utils for reading and writing TFRecords.

From  https://www.tensorflow.org/tutorials/load_data/tf_records#writing_a_tfrecord_file_2
"""
import numpy as np
import tensorflow as tf

def serialize_w_labels(patient, label, demo, row_mask, patient_t, args=args):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    ex.context.feature["patient_t"].int64_list.value.append(patient_t)
    ex.context.feature["n_patients"].int64_list.value.append(args.n_patients)
    ex.context.feature["max_t"].int64_list.value.append(args.max_t)
    ex.context.feature["max_v"].int64_list.value.append(args.max_v)
    # Feature lists for the two sequential features of our example
    fl_patients = ex.feature_lists.feature_list["patient"]
    fl_labels = ex.feature_lists.feature_list["label"]
    fl_demo = ex.feature_lists.feature_list["demo"]
    fl_row_masks = ex.feature_lists.feature_list["row_mask"]
    for visit, lab, dem, mask in zip(patient, label, demo, row_mask):
        fl_patients.feature.add().int64_list.value.append(visit)
        fl_labels.feature.add().int64_list.value.append(lab)
        fl_demo.feature.add().float_list.value.append(dem)
        fl_row_masks.feature.add().int64_list.value.append(mask)
    return ex.SerializeToString()