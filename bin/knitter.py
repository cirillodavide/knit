# usage:
# python bin/knitter.py --input_file1 out/coexpression.ecoli.511145_dist.csv --input_file2 out/ppi.ecoli.511145_rajagopala_dist.csv --tag coexpr

from __future__ import division
import tensorflow as tf
import pandas as pd
import numpy as np
import csv

tf.app.flags.DEFINE_string("input_file1", "NA", "")
tf.app.flags.DEFINE_string("input_file2", "NA", "")
tf.app.flags.DEFINE_string("tag", "NA", "")
FLAGS = tf.app.flags.FLAGS

tag = FLAGS.tag

filename1 = FLAGS.input_file1
dm1 = pd.read_csv(filename1,sep=',',skipinitialspace=True,index_col=0)
filename2 = FLAGS.input_file2
dm2 = pd.read_csv(filename2,sep=',',skipinitialspace=True,index_col=0)

intsect = list(set(dm1.index).intersection(dm2.index))
id_dict = dict()
for word in intsect:
		id_dict[word] = len(intsect)

def mynext (index, distMat):
	distMat = np.array(distMat)
	array = distMat[index,]
	order = array.argsort()
	ranks = order.argsort()
	ranks = map(lambda x: (x), ranks)
	return ranks.index(1)

writer = csv.writer(open("out/"+tag+"_knitout.csv", 'w'))
writer.writerow(("from","to","weight","type","gene"))
for i in intsect:
	cnt = 0
	cache = set()
	tmp = []
	tot = []
	k = i
	while(len(tmp)==0):
		if k in cache:
			tmp.append(k)
			break
		if k not in intsect:
			break
		cache.add(k)
		k_dm1_idx = dm1.columns.get_loc(k)
		j_dm1_idx = mynext(k_dm1_idx, dm1)
		j = dm1.iloc[j_dm1_idx].name
		tot.append([ k, j, cnt, 'obo1', i ])
		cnt += 1
		if j in cache:
			tmp.append(j)
			break
		if j not in intsect:
			break
		cache.add(j)
		j_dm2_idx = dm2.columns.get_loc(j)
		k_dm2_idx = mynext(j_dm2_idx, dm2)
		k = dm2.iloc[k_dm2_idx].name
		tot.append([ j, k, cnt, 'obo2', i ])
		cnt += 1
	for x in tot:
		x[2] = format((cnt - x[2])/cnt, '.3f')
		writer.writerow(x)
