"""Convert seqs and labels files to dense tensors and save to TFRecords.

The tensorflow implementation requires dense tensors. In order to be
able to use the list of list format from the original implementation's
process_mimic.py output, we convert the files here.

Since there is no information about the format of the demographics file
in the original implementation, it is assumed to have the same format as
the seqs and labels files, with one demo vector per visit, separated by
[-1].

Outputs zipped database of patient, patient_labs, visit_counts, patient_demos

A patient is a dense tensor [max_t, max_v].

Example of dense tensor conversion:
    seqs = [[1,2],[-1],[3,4],[5,6]]
    max_t = 3
    max_v = 4
    patients = tf.constant(np.array([[[1,2,-2,-2],
                                      [-2,-2,-2,-2]
                                      [-2,-2,-2,-2],
                                     [[3,4,-2,-2]
                                      [5,6,-2,-2]
                                      [-2,-2,-2,-2]]))
"""
import tensorflow as tf
import pickle
import argparse


parser = argparse.ArgumentParser(description='Convert seqs and labels'
                                 ' files to dense tensors and then into'
                                 ' TFRecord files for efficient batching'
                                 ' in med2vec-tf.')
parser.add_argument('--max_v', type=int,
                    help='Maximum value of |V|. That is, the most codes'
                    ' appearing in any single visit. Note that no visit'
                    ' in any of train, test, or predict should have'
                    ' more than max_v codes.', default=10)
parser.add_argument('--max_t', type=int,
                    help='The maximum value of |T|. That is, the most'
                    ' visits in any patients\'s record. Note that no'
                    ' patient in any of train, test, or predict should'
                    ' have more than max_t visits.')
parser.add_argument('--seqs_file', type=str,
                    help='Path to sequences file. Sequences file should'
                    ' be list of list. See README of original'
                    ' implementation.', default='seqs')
parser.add_argument('--labels_file', type=str,
                    help='Path to labels file. Labels file should'
                    ' be list of list. See README of original'
                    ' implementation.')
parser.add_argument('--demo_file', type=str,
                    help='Path to the demographics file.')
parser.add_argument('--out_file', type=str,
                    help='Path to output file.', default='zipped_TFR')
parser.add_argument('--n_patients', type=int,
                    help='Batch size.')
args = parser.parse_args()
args_dict = vars(args)


def load_data(args=args):
    """Replace later with dataset stuff."""
    seqs_file = args.seqs_file
    seqs = pickle.load(open(seqs_file, 'rb'))
    labs = None
    if args.labels_file is not None:
        labels_file = args.labels_file
        labs = pickle.load(open(labels_file, 'rb'))
    demo = None
    demo_dim = 0
    if args.demo_file is not None:
        demo_file = args.demo_file
        D_t = pickle.load(open(demo_file, 'rb'))
        demo_dim = D_t.shape[-1]
        demo = tf.constant(D_t, dtype=tf.float32)
        demo = tf.reshape(demo, [args.n_patients, -1])
    return seqs, labs, demo, demo_dim


def fill_visit(visit, args=args):
    """Fill all deficit visits with -2.

    Ensure that all visits have the same number of ICDs for efficient
    tensor logic. If a visit has fewer ICDs, filler ICDs get one-hot
    encoded as the zero vector, so that they affect nothing.

    visit: a list of integer medical codes

    Note: No visit in training or testing should have more than max_v
          visits.
    """
    max_v = args.max_v
    if visit != [-1]:
        new_visit = []
        new_visit.extend(visit)
        n_icd = len(visit)
        deficit = max_v - n_icd
        new_visit.extend([-2] * deficit)
        return new_visit


def fill_patient(patient, mask_batch, args=args):
    """Ensure that all patients have max_t visits.

    Create visits full of -2s, which are one-hot encoded as zero
    vectors. This makes all patients commensurate for efficient tensor
    logic.

    patient: list of list of integer codes
    max_t: the number of visits all patients ought to have

    Note: No patient in training or test data should have more
          than max_t visits.
    """
    max_t = args.max_t
    max_v = args.max_v
    new_patient = []
    new_patient.extend(patient)
    new_mask_batch = mask_batch
    t = len(new_patient)
    deficit = (max_t - t)
    new_patient.extend([[-2] * max_v] * deficit)
    new_mask_batch.append([[0] * max_v] * deficit)
    return new_patient, new_mask_batch, t


def tensorize_seqs(seqs, args=args, true_seqs=False):
    """Convert med2vec to tensorflow data.

    seqs: list of list. cf  https://github.com/mp2893/med2vec
    true_seqs: bool. Are we tensorizing the true sequences? If false,
               we are tonsorizing labels.
    returns:
        patients: tensor with shape [patients, max_t, max_v]

        row_masks: numpy array with shape [patients, max_t, max_v]
               Later, we will create a [patients, max_t, max_v, |C|]
               tensor where the [p, t, i, j] entry is p(c_j|c_i).
               Row_masks will drop the rows where c_i is the zero
               vector--that is, an NA ICD.

               A separate mask, col_mask, will be created from
               patients in order to mask, for each t, those j for
               which c_j did not appear in visit t, as well as
               p(c_i|c_i).

               The masks are to be applied in reverse order of creation.
               col_mask is applied with tf.multiply and row_masks
               with tf.boolean_mask to avoid needless reshaping.
        patients_ts: numpy array with shape [patients,] containing the
                     number of true visits for each patient.
    """
    patients = []
    new_patient = []
    row_masks = []
    mask_batch = []
    patients_ts = []
    for visit in seqs + [[-1]]:
        if visit != [-1]:
            visit = fill_visit(visit, args)
            new_patient.append(visit)
        else:
            new_patient, mask_batch, t = fill_patient(new_patient,
                                                      mask_batch,
                                                      args)
            patients.append(new_patient)
            if true_seqs:
                patients_ts.append(t)
                row_masks.append(mask_batch)
                mask_batch = []
            new_patient = []
    patients = tf.constant(patients)
    patients_ts = patients_ts
    row_masks = tf.not_equal(patients, -2)
    row_masks = tf.cast(row_masks, tf.int32)
    patients = tf.reshape(patients, [args.n_patients, -1])
    row_masks = tf.reshape(row_masks, [args.n_patients, -1])
    return patients, row_masks, patients_ts


# It feels a little ugly to have four versions of each function, but
# the alternative would be to map one function that checks for labels
# and demo inside of it, which would slow things down needlessly.
def serialize_lab_dem(patient, label, demo, row_mask, patient_t, args=args):
    """Turn each row of zipped dataset to example protos for writing to TFR."""
    ex = tf.train.SequenceExample()
    # Non-sequential features of the Example
    ex.context.feature["patient_t"].float_list.value.append(patient_t)
    ex.context.feature["max_t"].int64_list.value.append(args.max_t)
    ex.context.feature["max_v"].int64_list.value.append(args.max_v)
    ex.context.feature["demo_dim"].int64_list.value.append(args.demo_dim)
    # Feature lists for the "sequential" features of the Example
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


def serialize_lab(patient, label, row_mask, patient_t, args=args):
    """Turn each row of zipped dataset to example protos for writing to TFR."""
    ex = tf.train.SequenceExample()
    # Non-sequential features of the Example
    ex.context.feature["patient_t"].int64_list.value.append(patient_t)
    ex.context.feature["max_t"].int64_list.value.append(args.max_t)
    ex.context.feature["max_v"].int64_list.value.append(args.max_v)
    # Feature lists for the "sequential" features of the Example
    fl_patients = ex.feature_lists.feature_list["patient"]
    fl_labels = ex.feature_lists.feature_list["label"]
    fl_row_masks = ex.feature_lists.feature_list["row_mask"]
    for visit, lab, mask in zip(patient, label, row_mask):
        fl_patients.feature.add().int64_list.value.append(visit)
        fl_labels.feature.add().int64_list.value.append(lab)
        fl_row_masks.feature.add().int64_list.value.append(mask)
    return ex.SerializeToString()


def serialize_dem(patient, demo, row_mask, patient_t, args=args):
    """Turn each row of zipped dataset to example protos for writing to TFR."""
    ex = tf.train.SequenceExample()
    # Non-sequential features of the Example
    ex.context.feature["patient_t"].int64_list.value.append(patient_t)
    ex.context.feature["max_t"].int64_list.value.append(args.max_t)
    ex.context.feature["max_v"].int64_list.value.append(args.max_v)
    ex.context.feature["demo_dim"].int64_list.value.append(args.demo_dim)
    # Feature lists for the "sequential" features of the Example
    fl_patients = ex.feature_lists.feature_list["patient"]
    fl_labels = ex.feature_lists.feature_list["demo"]
    fl_row_masks = ex.feature_lists.feature_list["row_mask"]
    for visit, dem, mask in zip(patient, demo, row_mask):
        fl_patients.feature.add().int64_list.value.append(visit)
        fl_labels.feature.add().float_list.value.append(dem)
        fl_row_masks.feature.add().int64_list.value.append(mask)
    return ex.SerializeToString()


def serialize(patient, row_mask, patient_t, args=args):
    """Turn each row of zipped dataset to example protos for writing to TFR."""
    ex = tf.train.SequenceExample()
    # Non-sequential features of the Example
    ex.context.feature["patient_t"].int64_list.value.append(patient_t)
    ex.context.feature["max_t"].int64_list.value.append(args.max_t)
    ex.context.feature["max_v"].int64_list.value.append(args.max_v)
    # Feature lists for the "sequential" features of the Example
    fl_patients = ex.feature_lists.feature_list["patient"]
    fl_row_masks = ex.feature_lists.feature_list["row_mask"]
    for visit, mask in zip(patient, row_mask):
        fl_patients.feature.add().int64_list.value.append(visit)
        fl_row_masks.feature.add().int64_list.value.append(mask)
    return ex.SerializeToString()


def tf_serialize_lab_dem(patient, label, demo, row_mask, patient_t):
    """Map serialize_with_labels to tf.data.Dataset."""
    tf_string = tf.py_func(serialize_lab_dem,
                           (patient, label, demo, row_mask, patient_t),
                           tf.string)
    return tf.reshape(tf_string, ())


def tf_serialize_lab(patient, label, row_mask, patient_t):
    """Map serialize_with_labels to tf.data.Dataset."""
    tf_string = tf.py_func(serialize_lab,
                           (patient, label, row_mask, patient_t),
                           tf.string)
    return tf.reshape(tf_string, ())


def tf_serialize_dem(patient, demo, row_mask, patient_t):
    """Map serialize_with_labels to tf.data.Dataset."""
    tf_string = tf.py_func(serialize_dem,
                           (patient, demo, row_mask, patient_t),
                           tf.string)
    return tf.reshape(tf_string, ())


def tf_serialize(patient, row_mask, patient_t):
    """Map serialize_with_labels to tf.data.Dataset."""
    tf_string = tf.py_func(serialize,
                           (patient, row_mask, patient_t),
                           tf.string)
    return tf.reshape(tf_string, ())


if __name__ == '__main__':
    sess = tf.Session()
    seqs, labs, demo, args_dict['demo_dim'] = load_data()
    print("Labs is none:", labs is None, "Demo is none:", demo is None)
    data = []

    patients, row_masks, patients_ts = tensorize_seqs(seqs, true_seqs=True)
    data.append(patients)
    if labs is not None:
        print("args.labels_file is not None")
        labels, _, _ = tensorize_seqs(labs, true_seqs=False)
        data.append(labels)
        if demo is not None:
            print("################# LAB_DEM")
            data.append(demo)
            map_func = tf_serialize_lab_dem
        else:
            print("################ LAB")
            map_func = tf_serialize_lab
    elif demo is not None:
        print("################# DEM")
        data.append(demo)
        map_func = tf_serialize_dem
    else:
        print("################# NONE")
        map_func = tf_serialize
    data.extend([row_masks, patients_ts])

    output = tf.data.Dataset().from_tensor_slices(tuple(data))

    del data

    serialized = output.map(map_func)

    # Deprecated. Use tf.data.experimental.TFRecordWriter(...)
    writer = tf.contrib.data.TFRecordWriter(args.out_file)
    writeop = writer.write(serialized)
    sess.run(writeop)
    print("Done.")

