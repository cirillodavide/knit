# usage: python bin/factorizer.py --n_comp 5 --learn_rate 0.001 --penality 0.1 --input_file data/E.coli/coexpresion.ecoli.511145 --tag coexpr

import pandas as pd
import numpy as np
import sys
import os
from os.path import basename
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix
import tensorflow as tf
import scipy
from scipy import spatial
import csv

#======
# flags
#======

tf.app.flags.DEFINE_string("tag", "1", "")
tf.app.flags.DEFINE_string("input_file", "1", "")
tf.app.flags.DEFINE_integer("n_comp", "1", "")
tf.app.flags.DEFINE_float("learn_rate", "1", "")
tf.app.flags.DEFINE_float("penality", "1", "")

FLAGS = tf.app.flags.FLAGS

tag = FLAGS.tag
try:
	os.remove("out/"+tag+"_performances.csv")
except OSError:
    pass

#============
# data import
#============

filename = FLAGS.input_file

df = pd.read_csv(filename, sep='\t', engine='python')
df.columns = ["user", "item", "rate"]

if filename == 'data/E.coli/coexpression.ecoli.511145':
	df["user"] = df["user"].map(lambda x: x.lstrip("511145."))
	df["item"] = df["item"].map(lambda x: x.lstrip("511145."))
	
	# add inverted user-item pairs
	cols = list(df)
	cols[1], cols[0] = cols[0], cols[1]
	df1 = df.ix[:,cols]
	df1.columns = ["user", "item", "rate"]
	df = df.append(df1, ignore_index=True)
	
	# add 1.0 to same user-item pairs
	#df2 = pd.DataFrame()
	#df2['user'] = list(set(df['user']))
	#df2['item'] = list(set(df['user']))
	#df2['rate'] = [1.0] * len(set(df['user']))
	#df = df.append(df2, ignore_index=True)

	#x = df["rate"]
	#C = 1
	#D = 5
	#df["rate"] = C + (x - min(x)) * (D - C) / (max(x) - min(x))

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
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
	w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
	embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
	embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

	w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
	w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
	bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
	bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
	#bias_global = tf.get_variable("bias_global", shape=[])
	bias_global = tf.constant(mu, dtype=tf.float32, shape=[], name="bias_global")

with tf.variable_scope('Regularization'):
	penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="lambda")
	regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
	regularizer = tf.add(regularizer, tf.nn.l2_loss(bias_user))
	regularizer = tf.add(regularizer, tf.nn.l2_loss(bias_item), name="svd_regularizer")

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
		train_corr = scipy.stats.pearsonr(pred_batch, train_data) # pearson
		train_cosSim = 1 - spatial.distance.cosine(pred_batch, train_data) # cosSim
		
		pred_batch = sess.run(infer, feed_dict={user_batch: test_rows,
                                                item_batch: test_cols})
		test_err = np.sqrt(np.mean(np.power(pred_batch - test_data, 2)))
		test_corr = scipy.stats.pearsonr(pred_batch, test_data)
		test_cosSim = 1 - spatial.distance.cosine(pred_batch, test_data)
		
		if (i % 200) == 0:
			fields = [ i, train_err, train_cosSim, train_corr[0], test_err, test_cosSim, test_corr[0] ]
			print fields
			with open(r"out/"+tag+"_performances.csv", "a") as f:
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

print "global bias: ", b_g

estimates = []
user_factors = []
user_bias = []
item_factors = []
item_bias = []
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
		user_bias.append([ id_u, b_u[index] ])
	if id_i not in seen_i:
		seen_i.append(id_i)
		item_factors.append( q_i[index] )
		item_bias.append([ id_i, b_i[index] ])

tmp = pd.DataFrame(estimates, columns=[ 'user', 'item', 'rate', 'rate_estimate', 'user_bias', 'item_bias'])
tmp.to_csv('out/'+tag+'_estimates.csv', index=False, header=True)

tmp = pd.DataFrame(user_factors)
tmp.to_csv('out/'+tag+'_user-factors.csv', index=False, header=False)

tmp = pd.DataFrame(user_bias)
tmp.to_csv('out/'+tag+'_user-bias.csv', index=False, header=False)

tmp = pd.DataFrame(item_factors)
tmp.to_csv('out/'+tag+'_item-factors.csv', index=False, header=False)

tmp = pd.DataFrame(item_bias)
tmp.to_csv('out/'+tag+'_item-bias.csv', index=False, header=False)

from sklearn.metrics.pairwise import *
user_factors = pd.read_csv('out/'+tag+'_user-factors.csv',header=None)
user_factors = np.array(user_factors)
sim = cosine_similarity(user_factors)
simMat = pd.DataFrame(sim, index=seen_u, columns=seen_u)
simMat.to_csv("out/"+tag+"_user-similarity.csv")

item_factors = pd.read_csv('out/'+tag+'_item-factors.csv',header=None)
item_factors = np.array(item_factors)
sim = cosine_similarity(item_factors)
simMat = pd.DataFrame(sim, index=seen_i, columns=seen_i)
simMat.to_csv("out/"+tag+"_item-similarity.csv")


dm1 = pd.read_csv('out/'+tag+'_user-bias.csv',sep=',',index_col=0,header=None)
dm2 = pd.read_csv('out/'+tag+'_item-bias.csv',sep=',',index_col=0,header=None)

mat = []
for i in dm1.index:
	tensor = []
	b_u = dm1.ix[i,1]
	for j in dm2.index:
		b_i = dm2.ix[j,1]
		tensor.append(b_g + b_u + b_i)
	mat.append(tensor)

rMat = pd.DataFrame(mat, index=dm1.index, columns=dm2.index)
rMat.to_csv("out/"+tag+"_recommendations.csv")
