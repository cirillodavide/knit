# usage:
# python bin/knitter.py --input_file1 out/item-similarity.csv --input_file2 out/user-similarity.csv

from __future__ import division
import tensorflow as tf
import pandas as pd
import numpy as np
import csv

tf.app.flags.DEFINE_string("input_file1", "NA", "")
tf.app.flags.DEFINE_string("input_file2", "NA", "")
FLAGS = tf.app.flags.FLAGS

filename1 = FLAGS.input_file1
dm1 = pd.read_csv(filename1,sep=',',skipinitialspace=True,index_col=0)
filename2 = FLAGS.input_file2
dm2 = pd.read_csv(filename1,sep=',',skipinitialspace=True,index_col=0)

intsect = list(set(dm1.index).intersection(dm2.index))
id_dict = dict()
for word in intsect:
		id_dict[word] = len(intsect)

dm1 = np.array(dm1)
dm2 = np.array(dm2)

def mynext (index, distMat):
	array = distMat[index,]
	order = array.argsort()
	ranks = order.argsort()
	ranks = map(lambda x: (x), ranks)
	return ranks.index(1)

writer = csv.writer(open("out/knitout.csv", 'w'))
writer.writerow(("from","to","weight","type","gene"))
for i in range(len(intsect)):
	cnt = 0
	cache = set()
	tmp = []
	tot = []
	k = i
	while(len(tmp)==0):
		if k in cache:
			tmp.append(k)
			break
		cache.add(k)
		j = mynext(k, dm1)
		cnt += 1
		tot.append([intsect[k], intsect[j], cnt, 'obo1', intsect[i]])
		if j in cache:
			tmp.append(j)
			break
		cache.add(j)
		k = mynext(j, dm2)
		cnt += 1
		tot.append([intsect[j], intsect[k], cnt, 'obo2', intsect[i]])
	for x in tot:
		x[2] = format((len(tot) - x[2])/len(tot), '.3f')
		writer.writerow(x)
