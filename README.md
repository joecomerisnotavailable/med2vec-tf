# med2vec-tf
Tensorflow implementations of https://github.com/mp2893/med2vec

## To run:

See readme for mp2893's implementation for correct form of seqs and labs files, or
check provided simu_ data files for examples.

### process_seqs.py:

Use process_seqs.py to turn the theano-friendly seqs, labels and demo files into
TFRecords. The provided simulated data files can be processed using

python process_seqs.py --max_v 5 --max_t 8 --n_patients 25 --out_file ./data/simu_seqs_TFR --seqs_file simu_seqs --demo_file simu_demo --labels_file simu_labs

Or, to process your own files, use

python process_seqs.py --help

for information.

### med2vec-tf.py:

med2vec-tf.py can be run on the simulated sample of 1000 patients with 12 distinct medical codes, 7 labels, maximum 5 codes per visit and 8 visits per patient using the following command:

python med2vec-tf.py --max_v 5 --max_t 8 --n_codes 12 --n_labels 12 --code_emb_dim 4 --visit_emb_dim 5 --log_eps 1e-6 --win 2 --n_epochs 200 --log_dir simu_logs --n_patients 25 --demo --labels

The sequences were created according to a distribution so that certain codes cluster with other codes. For example, a successful embedding should show 1 and 5 very close, with 6 comparatively 8 separated.

### Note:

- Written for tensorflow-gpu 1.10. If you have a later version, it is recommended that you use a virtual environment. Otherwise, see the commented line near the bottom of process_seqs.py for a necessary alteration to that file.