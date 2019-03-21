"""Med2Vec.

# TF code written by Joe Comer (joecomerisnotavailable@gmail.com)
# Original implementation in Theano: https://github.com/mp2893/med2vec
# Original paper: Multi-layer Representation Learning for Medical Concepts,
                  Choi, et al.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import os


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
                    help='Hyperparameter. To avoid taking log of zero.')
parser.add_argument('--win', type=int,
                    help='Half the number of surrounding visits to'
                    ' include in the calculation of the visit cost.'
                    ' Corresponds to w in the summation index in eqn. 2'
                    ' of the original paper.')
parser.add_argument('--n_epochs', type=int,
                    help='Number of epochs to train.')
parser.add_argument('--data_dir', type=str,
                    help='Path to TFRecord files.', default='data')
parser.add_argument('--demo', action='store_true',
                    help='Include this tag if TFRecords include'
                    ' demographic data.')
parser.add_argument('--labels', action='store_true',
                    help='Include this tag if TFRecords include labels.')

args = parser.parse_args()


def parse_lab_dem(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64),
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
    output_shape = tf.stack(output_shape)
    patient = tf.reshape(seq_parsed['patient'], output_shape)
    label = tf.reshape(seq_parsed['label'], output_shape)
    demo = tf.reshape(seq_parsed['demo'], output_shape)
    row_mask = tf.reshape(seq_parsed['row_mask'], output_shape)
    patient_t = tf.reshape(ctxt_parsed['patient_t'], [1, 1])
    return (patient, label, demo, row_mask, patient_t)


def parse_lab(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64),
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
    return (patient, label, row_mask, patient_t)


def parse_dem(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64),
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
    output_shape = tf.stack(output_shape)
    patient = tf.reshape(seq_parsed['patient'], output_shape)
    demo = tf.reshape(seq_parsed['demo'], output_shape)
    row_mask = tf.reshape(seq_parsed['row_mask'], output_shape)
    patient_t = tf.reshape(ctxt_parsed['patient_t'], [1, 1])
    return (patient, demo, row_mask, patient_t)


def parse(example_proto, args=args):
    """Prepare TFRecords for training."""
    ctxt_fts = {
        "patient_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_t": tf.FixedLenFeature([], dtype=tf.int64),
        "max_v": tf.FixedLenFeature([], dtype=tf.int64),
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
    return (patient, row_mask, patient_t)


def choose_parse_function(args=args):
    """Choose the parse function to decode TFRecords.

    It seems ugly to have four separate functions, but doing these
    checks here is more efficient than mapping a function that does the
    checks internally to avoid the extra code."""
    if args.labels and args.demo:
        parse_func = parse_lab_dem
    elif args.labels:
        parse_func = parse_lab
    elif args.demo:
        parse_func = parse_dem
    else:
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
    x_t = tf.reduce_sum(patients, axis=-2)
    x_t = tf.expand_dims(x_t, -2)
    x_t = tf.tile(x_t, [1, 1, max_v, 1])
    col_masks = x_t - patients
    return col_masks


def codes_cost(patients, row_masks, visit_counts, W_c, b_c, args=args):
    """Calculate the cost for the code embeddings."""
    W_c_prime = tf.nn.relu(W_c)

    # tf.matmul doesn't broadcast, and we need to keep these grouped by
    # visit, so we need to tile W_c to one copy for every (real or
    # dummy) visit.
    W_c_tiled = tf.expand_dims(W_c_prime, 0)
    W_c_tiled = tf.expand_dims(W_c_tiled, 0)
    W_c_tiled = tf.tile(W_c_tiled, [args.n_patients, args.max_t, 1, 1])

    # w_ij is a n_patients X max_t array of code_emb_dim X max_v
    # matrices whose columns are the representations of the codes
    # appearing in each visit in seqs
    w_ij = tf.matmul(W_c_tiled, patients, transpose_b=True)

    # We want a patients X visits X max_v array of code_emb_dim X 1
    # vectors which are the columns from w_ij.
    w_ij = tf.transpose(w_ij, [0, 1, 3, 2])

    w_ij_shape = [args.n_patients,
                  args.max_t,
                  args.max_v,
                  args.code_emb_dim,
                  1]
    w_ij = tf.reshape(w_ij, w_ij_shape)

    # tf.multiply will broadcast these columns to each column of W_c in
    # each tile of W_c_tiled
    pre_sum = tf.multiply(W_c_prime, w_ij)
    logits = tf.reduce_sum(pre_sum, -2)

    # Logits now has a n_patients X max_t array of max_v X n_codes
    # vectors whose i, jth element is the dot product of the code
    # embedding of code i (which appears in visit t) with code j
    # (which may or may not)

    # The probability of code j given that code i is in the same visit
    p_j_i = tf.nn.softmax(logits, -1)

    log_p_j_i = tf.log(p_j_i + args.log_eps)

    # Create mask, but don't use it yet. See docstring for col_masks
    col_mask = col_masks(patients, args)

    # non_norm because we haven't divided by the number of real visits
    # for each patient yet.
    non_norm_summands = tf.multiply(log_p_j_i, col_mask)

    # Now for each patient divide by number of real visits of that
    # patient.
    # Mask rows corresponding to NA ICDs and p_i_i's afterward to ensure
    # patient-by-patient division.
    visit_counts = tf.reshape(visit_counts, [args.n_patients, 1, 1, 1])
    summands_w_dummies = non_norm_summands / visit_counts
    summands = tf.boolean_mask(summands_w_dummies, row_masks)
    codes_cost_per_visit = tf.reduce_sum(summands, -1)

    # Final cost is the batch average per patient of each patient's
    # average per visit cost
    codes_cost = tf.reduce_mean(codes_cost_per_visit)
    return codes_cost


def predictions(x_ts, W_c, D_t, W_v, W_s, b_c, b_v, b_s, demo_dim, args=args):
    """Get hat{y}_t."""

    # We don't need to group by visit in this branch. We also don't need
    # to buffer patients with dummy visits.
    x_2d = tf.reshape(x_ts, [-1, args.n_codes])
    dummy_visit_mask = tf.minimum(tf.reduce_sum(x_2d, -1), 1)
    dummy_visit_mask = tf.reshape(dummy_visit_mask, [-1,])

    if D_t is not None:
        d_2d = tf.reshape(D_t, [-1, demo_dim])

    u_ts = tf.matmul(W_c, x_2d, transpose_b=True)
    u_ts = tf.add(u_ts, b_c)
    u_ts = tf.transpose(u_ts)

    # In order to store D_t as a tensor it will need to have
    # dummy visits just like x_ts does. This also ensures that
    # everything aligns correctly when we concatenate, here.
    # But after concatenating, we can ditch the dummy visits.
    full_vec = tf.concat([u_ts, d_2d], axis=-1)
    full_vec = tf.boolean_mask(full_vec, dummy_visit_mask)

    v_t = tf.matmul(W_v, full_vec, transpose_b=True)
    v_t = tf.add(v_t, b_v)
    v_t = tf.transpose(v_t)

    pre_soft = tf.matmul(W_s, v_t, transpose_b=True)
    pre_soft = tf.add(pre_soft, b_s)
    pre_soft = tf.transpose(pre_soft)

    y_2d = tf.nn.softmax(pre_soft, axis=-1)
    return y_2d


def visits_cost(labels, y_2d, visit_counts, args):
    """Calculate the visits cost.

    labels: If there is no labels file, the labels are just x_ts.
    y_2d:   Output of predictions.
    visit_counts: A tensor of the number of true visits for each patient.

    outputs: The scalar visits prediction cost.
    """

    # We'll add the x vectors within the window before taking the dot
    # product with \hat{y}_t. To do this, we need to use a sliding
    # window, and to make sure patients' sums don't gather terms
    # from other patients, we need to pad each patient
    x_pad = tf.pad(labels, [[0, 0], [args.win, args.win], [0, 0]])

    # Because different \hat{y}_t have different numbers of
    # neighboring x_t in their window, we can't really avoid passing
    # 1-x_ts through the same loop as x_ts by subtracting final_x_totals
    # from 2*win / visit_counts, say
    z_pad = 1. - x_pad

    # Note that this is a different mask than the one produced in predictions.
    visit_mask = tf.minimum(tf.reduce_sum(x_pad, -1), 1)
    visit_mask = tf.reshape(visit_mask, [-1,])

    # We need to flatten x_pad to do the window function, so divide each x
    # by the number of visits of that patient *first*.
    normed_x_pad = x_pad / tf.reshape(visit_counts, [args.n_patients, 1, 1])
    normed_z_pad = z_pad / tf.reshape(visit_counts, [args.n_patients, 1, 1])

    normed_x_pad_2d = tf.reshape(normed_x_pad, [-1, args.n_labels])
    normed_z_pad_2d = tf.reshape(normed_z_pad, [-1, args.n_labels])

    # Before we padded around each patient. Now pad around the entire
    # list of visits
    x_double_pad = tf.pad(normed_x_pad_2d, [[args.win, args.win], [0, 0]])
    z_double_pad = tf.pad(normed_z_pad_2d, [[args.win, args.win], [0, 0]])

    def loop_ops(win_start, totalx, totalz):
        """Slide window function.

        Add x_ts from surrounding visits together before
        taking the dot product with log(hat{y}).

        For passing to tf.while_loop
        """
        summandx = tf.slice(x_double_pad,
                            [win_start, 0],
                            normed_x_pad_2d.shape)
        summandz = tf.slice(z_double_pad,
                            [win_start, 0],
                            normed_z_pad_2d.shape)
        return (win_start - 1,
                tf.add(totalx, summandx),
                tf.add(totalz, summandz))

    win_start = 2 * args.win
    totalx = tf.zeros(normed_x_pad_2d.shape)
    totalz = tf.zeros(normed_z_pad_2d.shape)
    loop_cond = lambda win_start, totalx, totalz: tf.less(-1, win_start)
    loop_fn = lambda win_start, totalx, totalz: loop_ops(win_start, totalx, totalz)
    _, window_x_total, window_z_total = tf.while_loop(loop_cond,
                                                      loop_ops,
                                                      (win_start, totalx, totalz))

    # Subtract out x_{t+0}
    correct_x_totals_pad = tf.subtract(window_x_total, normed_x_pad_2d)
    correct_z_totals_pad = tf.subtract(window_z_total, normed_z_pad_2d)

    final_x_total = tf.boolean_mask(correct_x_totals_pad, visit_mask)
    final_z_total = tf.boolean_mask(correct_z_totals_pad, visit_mask)

    summandsx = tf.multiply(final_x_total, tf.log(y_2d + args.log_eps))
    summandsz = tf.multiply(final_z_total, tf.log(1. - y_2d + args.log_eps))

    sumx = tf.reduce_sum(summandsx)
    sumz = tf.reduce_sum(summandsz)

    visits_cost = tf.subtract(sumz, sumx)
    return visits_cost


def create_vars(demo_dim, args=args):
    """Define weight matrices and biases."""
    W_c = tf.Variable(tf.truncated_normal([args.code_emb_dim, args.n_codes],
                      mean=0.0,
                      stddev=1.0,
                      dtype=tf.float32
                                          )
                      )
    W_v = tf.Variable(tf.truncated_normal(
                      shape=[args.visit_emb_dim, args.code_emb_dim + demo_dim],
                      mean=0.0,
                      stddev=1.0,
                      dtype=tf.float32
                                          )
                      )
    W_s = tf.Variable(tf.truncated_normal([args.n_labels, args.visit_emb_dim],
                      mean=0.0,
                      stddev=1.0,
                      dtype=tf.float32
                                          )
                      )

    b_c = tf.Variable(tf.zeros([W_c.shape[0], 1], dtype=tf.float32))
    b_v = tf.Variable(tf.zeros([W_v.shape[0], 1], dtype=tf.float32))
    b_s = tf.Variable(tf.zeros([W_s.shape[0], 1], dtype=tf.float32))
    return W_c, W_v, W_s, b_c, b_v, b_s


if __name__ == '__main__':
    parse_function = choose_parse_function()
    _, _, filenames = os.walk(args.data_dir)
    training_files, holdout = train_test_split(filenames, 0.25)
    validation_files, test_files = train_test_split(holdout, 0.4)

    W_c, W_v, W_s, b_c, b_v, b_s = create_vars(demo_dim)

    patients, row_masks, visit_counts = tensorize_seqs(seqs)

    x_ts = tf.reduce_sum(patients, -2)

    if args.labels_file is not None:
        labels, _, _ = tensorize_seqs(labs, true_seqs=False)
        # The call to tf.minimum is because labels may not be unique in
        # visits like ICDs/medical codes are. For example, if labels are
        # based on CSS groupings.
        labels = tf.minimum(tf.reduce_sum(labels, -2), 1)
    else:
        labels = x_ts

    code_cost = codes_cost(patients, row_masks, visit_counts, W_c, b_c)
    y_2d = predictions(x_ts, W_c, D_t, W_v, W_s, b_c, b_v, b_s, demo_dim)
    visit_cost = visits_cost(labels, y_2d, visit_counts, args)

    cost = tf.add(code_cost, visit_cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        for ep in list(range(args.n_epochs)):
            sess.run(optimizer)

            if ep % 5 == 0:

                print(cost.eval())
