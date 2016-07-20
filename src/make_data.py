import cPickle as pickle 
import numpy as np 
import re
import os
import json

def clearn(s, max_size=800):
	s = re.sub("\s+", " ", s)
	tokens = (s.strip().split(" "))[:max_size]
	tokens = tokens + ["<padding>"]*(max_size-len(tokens))
	return " ".join(tokens)


def txt2example(path, y="1"):
	with open(path) as mf:
		tx = mf.read()
		return clearn(tx) + "	" + y


def docs2corpus(rootdir, y, dest):
	corpus = []
	for parent, dirnames, filenames in os.walk(rootdir):
		for filename in filenames:
			if filename[0] == ".":	#hidden files
				continue
			x_y = txt2example(parent + "/" + filename, y)
			corpus += x_y,
	with open(dest, "wb") as mf:
		for line in corpus:
			mf.write(line+"\n")
	return corpus

def get_corps():
	pos = docs2corpus("../data/pos", "1", "../data/corpus_pos")
	neg = docs2corpus("../data/neg", "0", "../data/corpus_neg")
	corpus = pos + neg
	np.random.shuffle(corpus)
	with open("../data/corpus", "wb") as mf:
		for line in corpus:
			mf.write(line+"\n")
	return corpus

def get_dict():
	corpus = get_corps()
	token2id = {"<padding>":0, "<unk>":1}
	token_count = {}

	for line in corpus:
		tx = (line.strip().split("	"))[0]
		tokens = tx.split(" ")
		for tk in tokens:
			token_count.setdefault(tk, 0)
			token_count[tk] += 1
	i = 2
	for k, v in token_count.items():
		if k not in token2id and v > 5:
			token2id[k] = i
			i += 1

	with open("../data/token2id.json", "wb") as mf:
		json.dump(token2id, mf)

	return token2id

def get_id_corpus():
	corpus = get_corps()
	dix = get_dict()
	new_corpus = []
	for line in corpus:
		tx, label = line.strip().split("	")
		tokens = tx.split(" ")
		ids = []
		for tk in tokens:
			i = dix.get(tk, 1) #default to 1, <unk> (unknown)
			ids += str(i),
		ids_line = " ".join(ids) + "	" + label
		new_corpus += ids_line,
	with open("../data/idxcorpus", "wb") as mf:
		for line in new_corpus:
			mf.write(line+"\n")
	return new_corpus

def get_data(train_ratio):
	corpus = get_id_corpus()
	x, y = [], []
	for line in corpus:
		tx, label = line.strip().split("	")
		idxs = map(lambda x: int(x), tx.split(" "))
		x += idxs,
		y += int(label),
	train_size = int(len(x)*train_ratio)
	train_x = np.asarray(x[:train_size]).astype("int32")
	train_y = np.asarray(y[:train_size]).astype("int32")

	valid_x = np.asarray(x[train_size:]).astype("int32")
	valid_y = np.asarray(y[train_size:]).astype("int32")
	return (train_x, train_y), (valid_x, valid_y) 


if __name__ == '__main__':
	get_data(0.8)




