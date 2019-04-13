"""Med2Vec.

# TF code written by Joe Comer (joecomerisnotavailable@gmail.com)
# Original implementation in Theano: https://github.com/mp2893/med2vec
# Original paper: Multi-layer Representation Learning for Medical Concepts,
                  Choi, et al.
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import os
from time import time


import argparse


parser = argparse.ArgumentParser(description="Med2Vec.")
parser.add_argument('--n_patients', type=int,
                    help='Batch size.')
parser.add_argument('--max_v', type=int,
                    help='Maximum value of |V|. That is, the most codes'
                    ' appearing in any single visit. Note that no visit'
                    ' in any of train, test, or predict should have'
                    ' more than max_v codes.')
parser.add_argument('--max_t', type=int,
                    help='The maximum value of |T|. That is, the most'
                    ' visits in any patients\'s record. Note that no'
                    ' patient in any of train, test, or predict should'
                    ' have more than max_t visits.')
parser.add_argument('--n_codes', type=int,
                    help='The total number of unique medical codes.'
                    ' Note that this can be greater than the number'
                    ' of codes actually appearing in the data.')
parser.add_argument('--n_labels', type=int,
                    help='The total number of unique label codes.'
                    ' Note that this can be greater than the number'
                    ' of codes actually appearing in the data.')
parser.add_argument('--code_emb_dim', type=int,
                    help='The dimension of the medical code embedding.')
parser.add_argument('--visit_emb_dim', type=int,
                    help='The size of the visit embedding dimension.')
parser.add_argument('--log_eps', type=float,
                    help='Hyperparameter. To avoid taking log of zero.'
                    'defaule=1e-6', default=1e-6)
parser.add_argument('--win', type=int,
                    help='Half the number of surrounding visits to'
                    ' include in the calculation of the visit cost.'
                    ' Corresponds to w in the summation index in eqn. 2'
                    ' of the original paper.')
parser.add_argument('--n_epochs', type=int,
                    help='Number of epochs to train.')
parser.add_argument('--root_dir', type=str,
                    help='Path to root directory.', default='./')
parser.add_argument('--data_dir', type=str,
                    help='Path to TFRecord files.', default='data')
parser.add_argument('--demo', action='store_true',
                    help='Include this tag if TFRecords include'
                    ' demographic data.')
parser.add_argument('--labels', action='store_true',
                    help='Include this tag if TFRecords include labels.')
parser.add_argument('--log_dir', type=str, help='Directory in which to '
                    'store log data. default="logs"', default="logs")
parser.add_argument('--restore_checkpoint', action='store_true', help='Whether'
                    ' to continue from previously saved graph. default=False')
parser.add_argument('--checkpoint_dir', type=str, help='Path to saved graph '
                    'information. default="checkpoints"', default='checkpoints'
                    )

args = parser.parse_args()
args_dict = vars(args)


if not os.path.exists(os.path.join(args.root_dir, args.log_dir, "training")):
    os.makedirs(os.path.join(args.root_dir, args.log_dir, "training"))

if not os.path.exists(os.path.join(args.root_dir, args.log_dir, "validation")):
    os.makedirs(os.path.join(args.root_dir, args.log_dir, "validation"))


def h_m_s(time_delta):
    """Convert seconds to hours, minutes, seconds string format."""
    hours, r = divmod(int(time_delta), 3600)
    minutes, seconds = divmod(r, 60)
    return '{h}:{m:02d}:{m:02d}'.format(h=hours, m=minutes, s=seconds)


def printprogressbar(it,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=40,
                     fill='█'):
    """
    Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (it / float(total)))
    filledlength = int(length * it // total)
    bar = fill * filledlength + '-' * (length - filledlength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if it == total:
        print()


def parse_lab_dem(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.float32),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64),
        "demo_dim": tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fts = {
        "patient": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "label": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "demo": tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "row_mask": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    ctxt_parsed, seq_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=ctxt_fts,
        sequence_features=seq_fts
    )
    output_shape = [ctxt_parsed['max_t'], ctxt_parsed['max_v']]
    demo_shape = [ctxt_parsed['max_t'], ctxt_parsed['demo_dim']]
    output_shape = tf.stack(output_shape)
    demo_shape = tf.stack(demo_shape)
    patient = tf.reshape(seq_parsed['patient'], output_shape)
    label = tf.reshape(seq_parsed['label'], output_shape)
    demo = tf.reshape(seq_parsed['demo'], demo_shape)
    row_mask = tf.reshape(seq_parsed['row_mask'], output_shape)
    patient_t = tf.reshape(ctxt_parsed['patient_t'], [1, 1])
    return {'patient': patient, 'label': label, 'demo': demo,
            'row_mask': row_mask, 'patient_t': patient_t}


def parse_lab(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.float32),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fts = {
        "patient": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "label": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "row_mask": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    ctxt_parsed, seq_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=ctxt_fts,
        sequence_features=seq_fts
    )
    output_shape = [ctxt_parsed['max_t'], ctxt_parsed['max_v']]
    output_shape = tf.stack(output_shape)
    patient = tf.reshape(seq_parsed['patient'], output_shape)
    label = tf.reshape(seq_parsed['label'], output_shape)
    row_mask = tf.reshape(seq_parsed['row_mask'], output_shape)
    patient_t = tf.reshape(ctxt_parsed['patient_t'], [1, 1])
    return {'patient': patient, 'label': label,
            'row_mask': row_mask, 'patient_t': patient_t}


def parse_dem(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.float32),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64),
        "demo_dim": tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fts = {
        "patient": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "demo": tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "row_mask": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    ctxt_parsed, seq_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=ctxt_fts,
        sequence_features=seq_fts
    )
    output_shape = [ctxt_parsed['max_t'], ctxt_parsed['max_v']]
    demo_shape = [ctxt_parsed['max_t'], ctxt_parsed['demo_dim']]
    output_shape = tf.stack(output_shape)
    demo_shape = tf.stack(demo_shape)
    patient = tf.reshape(seq_parsed['patient'], output_shape)
    demo = tf.reshape(seq_parsed['demo'], demo_shape)
    row_mask = tf.reshape(seq_parsed['row_mask'], output_shape)
    patient_t = tf.reshape(ctxt_parsed['patient_t'], [1, 1])
    return {'patient': patient, 'demo': demo,
            'row_mask': row_mask, 'patient_t': patient_t}


def parse(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.float32),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64)
    }
    seq_fts = {
        "patient": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "row_mask": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    ctxt_parsed, seq_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=ctxt_fts,
        sequence_features=seq_fts
    )
    output_shape = [ctxt_parsed['max_t'], ctxt_parsed['max_v']]
    output_shape = tf.stack(output_shape)
    patient = tf.reshape(seq_parsed['patient'], output_shape)
    row_mask = tf.reshape(seq_parsed['row_mask'], output_shape)
    patient_t = tf.reshape(ctxt_parsed['patient_t'], [1, 1])
    return {'patient': patient,
            'row_mask': row_mask, 'patient_t': patient_t}


def choose_parse_function(args=args):
    """Choose the parse function to decode TFRecords.

    It seems ugly to have four separate functions, but doing these
    checks here is more efficient than mapping a function that does the
    checks internally to avoid the extra code.
    """
    if args.labels and args.demo:
        print("LAB_DEM")
        parse_func = parse_lab_dem
    elif args.labels:
        print("LAB")
        parse_func = parse_lab
    elif args.demo:
        print("DEM")
        parse_func = parse_dem
    else:
        print("NO LAB OR DEM")
        parse_func = parse
    return parse_func


def col_masks(patients, args=args):
    """Create a mask to cover non-present ICDs.

    For each V_t, for each c_i in V_t,
    zero out those p(c_j|c_i) for which c_j is not
    in V_t or for which i==j.

    See doc string for tensorize_seqs.

    patients: [patients,max_t,max_v,|C|] tensor

    returns: a binary tensor with shape patients.shape
    """
    max_v = args.max_v
    x_t = tf.reduce_sum(patients, axis=-2, name="cm_xt")
    x_t = tf.expand_dims(x_t, -2, name="cm_xt")
    x_t = tf.tile(x_t, [1, 1, max_v, 1], name="cm_xt")
    col_masks = tf.subtract(x_t, patients, name="col_masks")
    return col_masks


def codes_cost(patients, row_masks, visit_counts, W_c, b_c, args=args):
    """Calculate the cost for the code embeddings."""
    with tf.name_scope("Codes_Cost"):
        visit_counts = tf.cast(visit_counts, tf.float32,
                               name="cc_visit_counts")
        W_c_prime = tf.nn.relu(tf.transpose(W_c), name="w_c_prime")

        # tf.matmul doesn't broadcast, and we need to keep these grouped by
        # visit, so we need to tile W_c to one copy for every (real or
        # dummy) visit.
        W_c_tiled = tf.expand_dims(W_c_prime, 0, name="w_c_tiled")
        W_c_tiled = tf.expand_dims(W_c_tiled, 0, name="w_c_tiled")
        W_c_tiled = tf.tile(W_c_tiled, [args.n_patients, args.max_t, 1, 1],
                            name="w_c_tiled")

        # w_ij is a n_patients X max_t array of code_emb_dim X max_v
        # matrices whose columns are the representations of the codes
        # appearing in each visit in seqs
        w_ij = tf.matmul(W_c_tiled, patients, transpose_b=True, name="w_ij")

        # We want a patients X visits X max_v array of code_emb_dim X 1
        # vectors which are the columns from w_ij.
        w_ij = tf.transpose(w_ij, [0, 1, 3, 2], name="w_ij")

        w_ij_shape = [args.n_patients,
                      args.max_t,
                      args.max_v,
                      args.code_emb_dim,
                      1]
        w_ij = tf.reshape(w_ij, w_ij_shape, name="w_ij")

        # tf.multiply will broadcast these columns to each column of W_c in
        # each tile of W_c_tiled
        pre_sum = tf.multiply(W_c_prime, w_ij, name="cc_pre_sum")
        logits = tf.reduce_sum(pre_sum, -2, name="cc_logits")

        # Logits now has a n_patients X max_t array of max_v X n_codes
        # vectors whose i, jth element is the dot product of the code
        # embedding of code i (which appears in visit t) with code j
        # (which may or may not)

        # The probability of code j given that code i is in the same visit
        p_j_i = tf.nn.softmax(logits, -1, name="cc_p_j_i")

        log_p_j_i = tf.log(p_j_i + args.log_eps, name="cc_log_p_j_i")

        # Create mask, but don't use it yet. See docstring for col_masks
        col_mask = col_masks(patients, args)

        # non_norm because we haven't divided by the number of real visits
        # for each patient yet.
        non_norm_summands = tf.multiply(log_p_j_i, col_mask,
                                        name="cc_non_norm_summands")

        # Now for each patient divide by number of real visits of that
        # patient.
        # Mask rows corresponding to NA ICDs and p_i_i's afterward to ensure
        # patient-by-patient division.
        visit_counts = tf.expand_dims(visit_counts, -1, name="cc_visit_counts")
        summands_w_dummies = non_norm_summands / visit_counts
        summands = tf.boolean_mask(summands_w_dummies, row_masks,
                                   name="summands")
        codes_cost_per_visit = tf.reduce_sum(summands, -1,
                                             name="codes_cost_per_visit")

        # Final cost is the batch average per patient of each patient's
        # average per visit cost
        codes_cost = tf.reduce_mean(codes_cost_per_visit, name="codes_cost")
        return codes_cost


def predictions(x_ts, W_c, D_t, W_v, W_s, b_c, b_v, b_s, demo_dim, args=args):
    """Get hat{y}_t."""
    with tf.name_scope("Make_Predictions"):
        # We don't need to group by visit in this branch. We also don't need
        # to buffer patients with dummy visits.
        x_2d = tf.reshape(x_ts, [-1, args.n_codes],
                          name="pred_x2d")
        dummy_visit_mask = tf.minimum(tf.reduce_sum(x_2d, -1), 1,
                                      name="pred_dummy_vm")
        dummy_visit_mask = tf.reshape(dummy_visit_mask, [-1,],
                                      name="pred_dummy_vm")

        if D_t is not None:
            d_2d = tf.reshape(D_t, [-1, demo_dim], name="pred_d_2d")
        else:
            d_2d = None

        u_ts = tf.matmul(x_2d, W_c,
                         name="pred_u_ts")
        u_ts = tf.transpose(u_ts)
        u_ts = tf.add(u_ts, b_c, name="pred_u_ts")
        u_ts = tf.transpose(u_ts, name="pred_u_ts")

        # In order to store D_t as a tensor it will need to have
        # dummy visits just like x_ts does. This also ensures that
        # everything aligns correctly when we concatenate, here.
        # But after concatenating, we can ditch the dummy visits.
        if d_2d is not None:
            full_vec = tf.concat([u_ts, d_2d], axis=-1, name="pred_full_vec")
        else:
            full_vec = u_ts
        full_vec = tf.boolean_mask(full_vec, dummy_visit_mask,
                                   name="pred_full_vec")

        v_t = tf.matmul(W_v, full_vec, transpose_b=True, name="pred_vt")
        v_t = tf.add(v_t, b_v, name="pred_vt")
        v_t = tf.transpose(v_t, name="pred_vt")

        pre_soft = tf.matmul(W_s, v_t, transpose_b=True, name="pred_pre_soft")
        pre_soft = tf.add(pre_soft, b_s, name="pred_pre_soft")
        pre_soft = tf.transpose(pre_soft, name="pred_pre_soft")

        y_2d = tf.nn.softmax(pre_soft, axis=-1, name="pred_y_2d")
        return y_2d


def visits_cost(labels, y_2d, visit_counts, args):
    """Calculate the visits cost.

    labels: If there is no labels file, the labels are just x_ts.
    y_2d:   Output of predictions.
    visit_counts: A tensor of the number of true visits for each patient.

    outputs: The scalar visits prediction cost.
    """
    with tf.name_scope("Visits_Cost"):
        visit_counts = tf.cast(visit_counts, tf.float32,
                               name="vc_visit_counts")
        # We'll add the x vectors within the window before taking the dot
        # product with \hat{y}_t. To do this, we need to use a sliding
        # window, and to make sure patients' sums don't gather terms
        # from other patients, we need to pad each patient
        x_pad = tf.pad(labels, [[0, 0], [args.win, args.win], [0, 0]],
                       name="x_pad")

        # Because different \hat{y}_t have different numbers of
        # neighboring x_t in their window, we can't really avoid passing
        # 1-x_ts through the same loop as x_ts by subtracting final_x_totals
        # from 2*win / visit_counts, say
        z_pad = tf.subtract(1., x_pad, name="vc_z_pad")

        # Note that this is a different mask than the one produced in
        # predictions.
        visit_mask = tf.minimum(tf.reduce_sum(x_pad, -1), 1,
                                name="vc_visit_mask")
        visit_mask = tf.reshape(visit_mask, [-1,], name="vc_visit_mask")

        # We need to flatten x_pad to do the window function, so divide each x
        # by the number of visits of that patient *first*.
        normed_x_pad = x_pad / tf.reshape(visit_counts,
                                          [args.n_patients, 1, 1])
        normed_z_pad = z_pad / tf.reshape(visit_counts,
                                          [args.n_patients, 1, 1])

        normed_x_pad_2d = tf.reshape(normed_x_pad, [-1, args.n_labels],
                                     name="vc_normed_x_pad_2d")
        normed_z_pad_2d = tf.reshape(normed_z_pad, [-1, args.n_labels],
                                     name="vc_normed_x_pad_2d")

        # Before we padded around each patient. Now pad around the entire
        # list of visits
        x_double_pad = tf.pad(normed_x_pad_2d, [[args.win, args.win], [0, 0]],
                              name="x_double_pad")
        z_double_pad = tf.pad(normed_z_pad_2d, [[args.win, args.win], [0, 0]],
                              name="z_double_pad")

        slice_height = args.n_patients * (args.max_t + args.win * 2)
        slice_shape = [slice_height, args.n_labels]

        def loop_ops(win_start, totalx, totalz):
            """Slide window function.

            Add x_ts from surrounding visits together before
            taking the dot product with log(hat{y}).

            For passing to tf.while_loop
            """
            tail_length = 2 * args.win - win_start
            parts = tf.concat([tf.zeros(win_start, dtype=tf.int32),
                               tf.ones(slice_height, dtype=tf.int32),
                               tf.zeros(tail_length, dtype=tf.int32)], -1)
            summandx = tf.dynamic_partition(x_double_pad,
                                            num_partitions=2,
                                            partitions=parts,
                                            name="summandx")[1]
            summandz = tf.dynamic_partition(z_double_pad,
                                            num_partitions=2,
                                            partitions=parts,
                                            name="summandz")[1]
            return (win_start - 1, tf.add(totalx, summandx),
                    tf.add(totalz, summandz))

        win_start = 2 * args.win

        totalx = tf.zeros(slice_shape, dtype=tf.float32, name="totalx")
        totalz = tf.zeros(slice_shape, dtype=tf.float32, name="totalz")
        loop_cond = lambda win_start, totalx, totalz: tf.less(-1, win_start)
        loop_fn = lambda win_start, totalx, totalz: loop_ops(win_start,
                                                             totalx,
                                                             totalz)
        _, window_x_total, window_z_total = tf.while_loop(loop_cond,
                                                          loop_ops,
                                                          (win_start,
                                                           totalx,
                                                           totalz),
                                                          name="while_loop")

        # Subtract out x_{t+0}
        correct_x_totals_pad = tf.subtract(window_x_total, normed_x_pad_2d,
                                           name="correct_x_totals_pad")
        correct_z_totals_pad = tf.subtract(window_z_total, normed_z_pad_2d,
                                           name="correct_z_totals_pad")

        final_x_total = tf.boolean_mask(correct_x_totals_pad, visit_mask,
                                        name="final_x_cost")
        final_z_total = tf.boolean_mask(correct_z_totals_pad, visit_mask,
                                        name="final_z_cost")

        summandsx = tf.multiply(final_x_total, tf.log(y_2d + args.log_eps),
                                name="summandsx")
        summandsz = tf.multiply(final_z_total,
                                tf.log(1. - y_2d + args.log_eps),
                                name="summandsz")

        sumx = tf.reduce_sum(summandsx, name="sumx")
        sumz = tf.reduce_sum(summandsz, name="sumz")

        visits_cost = tf.subtract(sumz, sumx, name="visits_cost")
        return visits_cost


def create_vars(demo_dim, args=args):
    """Define weight matrices and biases."""
    with tf.variable_scope("Embeddings"):
        W_c = tf.Variable(tf.truncated_normal([args.n_codes,
                                               args.code_emb_dim],
                          mean=0.0,
                          stddev=1.0,
                          dtype=tf.float32
                                              ),
                          name="W_c")
        W_v = tf.Variable(tf.truncated_normal(
                          shape=[args.visit_emb_dim,
                                 args.code_emb_dim + demo_dim],
                          mean=0.0,
                          stddev=1.0,
                          dtype=tf.float32),
                          name="W_v")
        W_s = tf.Variable(tf.truncated_normal([args.n_labels,
                                               args.visit_emb_dim],
                          mean=0.0,
                          stddev=1.0,
                          dtype=tf.float32
                                              ),
                          name="W_s")

        b_c = tf.Variable(tf.zeros([W_c.shape[1], 1], dtype=tf.float32),
                          name="b_c")
        b_v = tf.Variable(tf.zeros([W_v.shape[0], 1], dtype=tf.float32),
                          name="b_v")
        b_s = tf.Variable(tf.zeros([W_s.shape[0], 1], dtype=tf.float32),
                          name="b_s")
        return W_c, W_v, W_s, b_c, b_v, b_s


def get_demo_dim(filelist, parse_function, args=args):
    """Retrieve the demo vector dimension before the full graph runs."""
    with tf.name_scope("get_demo_dimension"):
        with tf.Session() as sess:
            temp_ds = tf.data.TFRecordDataset(filelist[0])
            temp_it = temp_ds.make_one_shot_iterator()
            serial = temp_it.get_next()
            sample = parse_function(serial)['demo']
            demo_dim = sess.run(sample).shape[-1]
    return demo_dim


if __name__ == '__main__':
    run_start = time()
    data_path = args.root_dir + args.data_dir
    parse_function = choose_parse_function()
    filelist = [os.path.join(data_path, filename)
                for filename in os.listdir(data_path)]
    if args.demo:
        demo_dim = get_demo_dim(filelist, parse_function)
    training_files, holdout = train_test_split(filelist,
                                               test_size=0.25)
    validation_files, test_files = train_test_split(holdout,
                                                    test_size=0.4)

    if len(training_files) == 0:
        training_files.append(filelist[0])
    if len(validation_files) == 0:
        validation_files.append(filelist[0])

    n_train_files = len(training_files)

    with tf.name_scope("Batch"):
        filenames = tf.placeholder(tf.string, shape=[None])

        data = tf.data.TFRecordDataset(filenames)
        data = data.map(parse_function)

        data = data.batch(args.n_patients)
        iterator = data.make_initializable_iterator()

        batch = iterator.get_next()

        patients = batch['patient']
        patients = tf.one_hot(patients, args.n_codes, name="one_hot_patients")
        if args.labels:
            labels = batch['label']
            labels = tf.one_hot(labels, args.n_labels, name="one_hot_labels")
        else:
            labels = patients
            args_dict['n_labels'] = args.n_codes
        if args.demo:
            demo = batch['demo']
        else:
            demo = None
            demo_dim = 0
        row_masks = batch['row_mask']
        visit_counts = batch['patient_t']

    W_c, W_v, W_s, b_c, b_v, b_s = create_vars(demo_dim)

    with tf.name_scope("Binary_visit_reps"):
        x_ts = tf.reduce_sum(patients, -2, name="global_x_ts")

    if args.labels:
        # The call to tf.minimum is because labels may not be unique
        # in visits like ICDs/medical codes are. For example, if
        # labels are based on CSS groupings.
        labels = tf.minimum(tf.reduce_sum(labels, -2), 1, name="agg_labels")
    else:
        labels = x_ts

    code_cost = codes_cost(patients, row_masks,
                           visit_counts, W_c, b_c)
    y_2d = predictions(x_ts, W_c, demo, W_v, W_s,
                       b_c, b_v, b_s, demo_dim)
    visit_cost = visits_cost(labels, y_2d, visit_counts, args)

    cost = tf.add(code_cost, visit_cost, name="cost")
    with tf.name_scope("Summaries"):
        summ_code_cost = tf.summary.scalar("Code_cost", code_cost)
        summ_visit_cost = tf.summary.scalar("Visit_cost", visit_cost)
        summ_cost = tf.summary.scalar("Total_cost", cost)

        merged = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(name="optimizer").minimize(cost)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path
                                             .join(args.root_dir,
                                                   args.log_dir,
                                                   "training"),
                                             sess.graph)
        valid_writer = tf.summary.FileWriter(os.path
                                             .join(args.root_dir,
                                                   args.log_dir,
                                                   "validation"),
                                             sess.graph)
        restoring = (tf.train.checkpoint_exists(args.checkpoint_dir) and
                     args.restore_checkpoint)
        if restoring:
            checkpoints = [int(f.split("_")[0])
                           for f in os.listdir(args.checkpoint_dir)
                           if len(f.split("_")) > 1]
            start_ep = max(checkpoints)
            saver.restore(sess,
                          os.path.join(args.checkpoint_dir,
                                       '{}_saved_model'
                                       .format(start_ep)))
        else:
            start_ep = -1
            sess.run(init)
        print_train = -0.0
        print_val = -0.0

        for ep in range(start_ep + 1, args.n_epochs):
            ep_start = time()
            sess.run(iterator.initializer,
                     feed_dict={filenames: training_files})
            try:
                while True:
                    sess.run(optimizer)
            except tf.errors.OutOfRangeError:
                pass
            sess.run(iterator.initializer,
                     feed_dict={filenames: training_files})
            summ = sess.run(merged)
            print_train = sess.run(cost)
            printprogressbar(ep,
                             args.n_epochs,
                             prefix='Ep {e}'.format(e=ep),
                             suffix='Ep {a} | All {b}'
                             .format(a=h_m_s(time() - ep_start),
                                     b=h_m_s(time() - run_start)
                                     ))
            train_writer.add_summary(summ, ep)
            # Initialize iterator with validation data
            if ep % 5 == 0:
                sess.run(iterator.initializer,
                         feed_dict={filenames: validation_files})
                summ = sess.run(merged)
                print_val = sess.run(cost)
                printprogressbar(ep,
                                 args.n_epochs,
                                 prefix='Ep {e}'.format(e=ep),
                                 suffix='Ep {a} | All {b}'
                                 .format(a=h_m_s(time() - ep_start),
                                         b=h_m_s(time() - run_start)
                                         ))

                valid_writer.add_summary(summ, ep)

                save_path = saver.save(sess,
                                       os.path.join(args.root_dir,
                                                    args.log_dir,
                                                    str(ep) + '_saved_model'))
        embedding_dict = {"W_c": sess.run(W_c),
                          "W_v": sess.run(W_v),
                          "W_s": sess.run(W_s),
                          "b_c": sess.run(b_c),
                          "b_v": sess.run(b_v),
                          "b_s": sess.run(b_s)}
        emb_path = os.path.join(args.root_dir + args.log_dir, "embeddings")
        with open(emb_path, 'wb') as emb_file:
            pickle.dump(embedding_dict, emb_file, protocol=2)
