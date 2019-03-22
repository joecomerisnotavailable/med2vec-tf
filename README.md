# med2vec-tf
Tensorflow implementations of https://github.com/mp2893/med2vec

Currently in the tuning and debugging phase. Prints only minimal output and does not yet save graph or embeddings.

Can be run on the dummy data TFRecords (serialized from the (very tiny) dummy data *seqs*, *labels* and *demo* files also provided) with the following command:

python med2vec-tf.py --n_patients 2 --max_v 6 --max_t 5 --n_codes 7 --n_labels 4 --code_emb_dim 4 --visit_emb_dim 5 --win 3 --n_epochs 10 --log_eps 1e-6 --demo --labels

