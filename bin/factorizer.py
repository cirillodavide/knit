# usage: python bin/factorizer.py --n_comp 5 --learn_rate 0.001 --penality 0.1 --input_file data/E.coli/coexpresion.ecoli.511145

import pandas as pd
import numpy as np
import sys
import os
from os.path import basename
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix
import tensorflow as tf
from scipy import spatial
import csv

try:
	os.remove("out/performances.csv")
except OSError:
    pass

try:
	os.remove("out/estimates.csv")
except OSError:
    pass

try:
	os.remove("out/user-factors.csv")
except OSError:
    pass

try:
	os.remove("out/item-factors.csv")
except OSError:
    pass

#======
# flags
#======

tf.app.flags.DEFINE_string("input_file", "1", "")
tf.app.flags.DEFINE_integer("n_comp", "1", "")
tf.app.flags.DEFINE_float("learn_rate", "1", "")
tf.app.flags.DEFINE_float("penality", "1", "")

FLAGS = tf.app.flags.FLAGS

#============
# data import
#============

filename = FLAGS.input_file

df = pd.read_csv(filename, sep='\t', engine='python')
df.columns = ["user", "item", "rate"]

if filename == 'data/E.coli/coexpresion.ecoli.511145':
	df["user"] = df["user"].map(lambda x: x.lstrip("511145."))
	df["item"] = df["item"].map(lambda x: x.lstrip("511145."))

if filename == 'data/E.coli/ppi.ecoli.511145':
	df["user"] = df["user"].map(lambda x: x.lstrip("511145."))
	df["item"] = df["item"].map(lambda x: x.lstrip("511145."))

	s = df["rate"]
	foo = lambda x: pd.Series([i for i in reversed(x.split('|'))])
	rev = s.apply(foo)
	rev.rename(columns={0:'rajagopala',1:'hu_revised',2:'hu',3:'arifuzzaman'},inplace=True)
	rev = rev[['arifuzzaman','hu','hu_revised','rajagopala']]

	var = raw_input("Choose set (arifuzzaman/hu/hu_revised/rajagopala): ")

	dset = str(var.strip())
	df = df.assign(rate=pd.Series(rev[dset]).values)
	df.drop(df.columns[[2]], axis=1)

#=======================
# training and test sets
#=======================

def matrix (dataframe):
	user_dict = dict()
	item_dict = dict()
	for word in set(dataframe["user"]):
		user_dict[word] = len(user_dict)
	for word in set(dataframe["item"]):
		item_dict[word] = len(item_dict)

	rows = np.array([ user_dict[x] for x in dataframe["user"] ])
	cols = np.array([ item_dict[x] for x in dataframe["item"] ])
	data = np.array(dataframe["rate"].astype(np.float32))

	num_rows = len(user_dict) 
	num_cols = len(item_dict)

	return rows, cols, data, num_rows, num_cols, user_dict, item_dict

rows, cols, data, num_rows, num_cols, user_dict, item_dict = matrix(df)

###

from sklearn import model_selection as cv
train_data_matrix, test_data_matrix = cv.train_test_split(df, test_size=0.10)

train_rows = np.array([ user_dict[x] for x in train_data_matrix["user"] ])
train_cols = np.array([ item_dict[x] for x in train_data_matrix["item"] ])
train_data = np.array(train_data_matrix["rate"].astype(np.float32))

test_rows = np.array([ user_dict[x] for x in test_data_matrix["user"] ])
test_cols = np.array([ item_dict[x] for x in test_data_matrix["item"] ])
test_data = np.array(test_data_matrix["rate"].astype(np.float32))

#======
# model
#======

user_num = num_rows
item_num = num_cols
dim = FLAGS.n_comp
learning_rate = FLAGS.learn_rate
reg = FLAGS.penality
mu = np.mean(data)
epochs = 10001

# objective definition

with tf.variable_scope('Placeholder'):
		user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
		item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
		rate_batch = tf.placeholder(tf.float32, shape=[None])

with tf.variable_scope('Estimates'):
	w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=1))
	w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=1))

	#bias_global = tf.get_variable("bias_global", shape=[])
	bias_global = tf.constant(mu, dtype=tf.float32, shape=[], name="bias_global")
	w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
	w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
	bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
	bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")

	embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
	embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

with tf.variable_scope('Regularization'):
	penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
	regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")

with tf.variable_scope('Inference'):
	infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
	infer = tf.add(infer, bias_global)
	infer = tf.add(infer, bias_user)
	infer = tf.add(infer, bias_item, name="svd_inference")

with tf.variable_scope('Cost'):
	cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
	cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))

estimates = [ bias_global, bias_user, bias_item ]
factors = [ embd_user, embd_item ] 

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# training loop

estimates_batch = []
factors_batch = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epochs):
		_, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: train_rows,
                                                               item_batch: train_cols,
                                                               rate_batch: train_data})
		train_err = np.sqrt(np.mean(np.power(pred_batch - train_data, 2))) # RMSE
		train_cosSim = 1 - spatial.distance.cosine(pred_batch, train_data)
		
		pred_batch = sess.run(infer, feed_dict={user_batch: test_rows,
                                                item_batch: test_cols})
		test_err = np.sqrt(np.mean(np.power(pred_batch - test_data, 2)))
		test_cosSim = 1 - spatial.distance.cosine(pred_batch, test_data)
		
		if (i % 200) == 0:
			fields = [ i, train_err, train_cosSim, test_err, test_cosSim ]
			print fields
			with open(r"out/performances.csv", "a") as f:
				writer = csv.writer(f)
				writer.writerow(fields)
		
	pred_batch, est_batch, fact_batch = sess.run([infer, estimates, factors], feed_dict={user_batch: rows,
                                                                    item_batch: cols})
	estimates_batch = est_batch
	factors_batch = fact_batch

#===============
# write to files
#===============

b_g = estimates_batch[0]
b_u = estimates_batch[1]
b_i = estimates_batch[2]
p_u = factors_batch[0]
q_i = factors_batch[1]

print "average rating: ", b_g

estimates = []
user_factors = []
item_factors = []
seen_u = []
seen_i = []
for index in range(len(data)):
	id_u = user_dict.keys()[user_dict.values().index(rows[index])]	
	id_i = item_dict.keys()[item_dict.values().index(cols[index])]
	rate_ui = data[index]
	ratePred_ui = b_g + b_u[index] + b_i[index]
	estimates.append([ id_u, id_i, rate_ui, ratePred_ui, b_u[index], b_i[index] ])
	if id_u not in seen_u:
		seen_u.append(id_u)
		user_factors.append( p_u[index] )
	if id_i not in seen_i:
		seen_i.append(id_i)
		item_factors.append( q_i[index] )

tmp = pd.DataFrame(estimates, columns=[ 'user', 'item', 'rate', 'rate_estimate', 'user_bias', 'item_bias'])
tmp.to_csv('out/estimates.csv', index=False, header=True)

tmp = pd.DataFrame(user_factors)
tmp.to_csv('out/user-factors.csv', index=False, header=False)

tmp = pd.DataFrame(item_factors)
tmp.to_csv('out/item-factors.csv', index=False, header=False)

from sklearn.metrics.pairwise import cosine_similarity


simMat = []
for i in range(len(seen_u)):
	for j in range(len(seen_u)):
		sim = cosine_similarity(user_factors[i], user_factors[j])
		print seen_u[i]
		print user_factors[i]
		print seen_u[j]
		print user_factors[j]
		print [ seen_u[i], seen_u[j], sim ]

'''
simMat = []
for i in range(len(seen_i)):
	for j in range(len(seen_i)):
		sim = 1 - spatial.distance.cosine(item_factors[i], item_factors[j])
		simMat.append(sim)
simMat = pd.DataFrame(sim, index=seen_i, columns=seen_i)
simMat.to_csv("out/item-similarity.csv")
'''