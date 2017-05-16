# usage: python bin/recommender.py --input_file out/item-factors.csv

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import csv

tf.app.flags.DEFINE_string("input_file", "1", "")
FLAGS = tf.app.flags.FLAGS

filename = FLAGS.input_file

df = pd.read_csv(filename, sep=',', engine='python', header=None, index_col=0)

from sklearn.metrics.pairwise import cosine_similarity
simMat = pd.DataFrame(cosine_similarity(df), index=df.index, columns=df.index)

simMat.to_csv("out/item-similarity.csv")



writer = csv.writer(open("knit.csv", 'w'))
writer.writerow(("from","to","weight","type","gene"))
for i in range(len(df.index)):
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
		j = mynext(k, obo1_dist)
		cnt += 1
		tot.append([intsect[k], intsect[j], cnt, 'obo1', intsect[i]])
		if j in cache:
			tmp.append(j)
			break
		cache.add(j)
		k = mynext(j, obo2_dist)
		cnt += 1
		tot.append([intsect[j], intsect[k], cnt, 'obo2', intsect[i]])
	for x in tot:
		x[2] = format((len(tot) - x[2])/len(tot), '.3f')
		writer.writerow(x)
