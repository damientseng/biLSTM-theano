from layers import *
import cPickle as pickle 
import make_data

import logging
logging.basicConfig(level=logging.DEBUG,
	format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
	datefmt='%a, %d %b %Y %H:%M:%S',
	filename='trainingprocess.log',
	filemode='w')

class BiLSTMQA(object):

	def __init__(self, voca_size, hidden_size, ydim, learn_rate=0.1):

		self.nnet = StackedBiLSTM(voca_size, hidden_size, ydim)

	def fit(self, x, y, vx=None, vy=None, max_epochs=10000, batch_size=5):
		
		mask = self.__get_mask(x)
		num_batches = x.shape[1] // batch_size
		batch_idx = 0
		for ep in range(max_epochs):
			btx = x[:, batch_idx*batch_size:(batch_idx+1)*batch_size]
			bty = y[batch_idx*batch_size:(batch_idx+1)*batch_size]
			btmask = mask[:, batch_idx*batch_size:(batch_idx+1)*batch_size]

			loss = self.nnet.train(btx, btmask, bty)

			if ep%20 == 0:
				print "in epoch %d/%d..."%(ep, max_epochs)
			if batch_idx == 0:
				ot = "in epoch %d/%d..."%(ep, max_epochs) + "	loss:	"+str(loss)
				print ot
				logging.info(ot)
				"""
				validate
				if vx != None:
					print self.score(vx, vy)
				"""
			batch_idx = (batch_idx+1) % num_batches


	def predict(self, x):
		mask = self.__get_mask(x)
		return self.nnet.predict(x, mask)

	def score(self, x, y):
		prd = self.predict(x)
		s = 0
		for i in range(len(y)):
			s += 1. if prd[i] == y[i] else 0.
		return s/len(y)


	def self_pickl(self, path="./data/bilstm_model.pkl"):
		with open(path, "wb") as mf:
			pickle.dump(self, mf)

	def __get_mask(self, data):
		mask = np.not_equal(data, 0).astype("int32")
		return mask 


if __name__ == '__main__':

	train, valid = make_data.get_data(0.9)
	train_x, train_y = train 
	valid_x, valid_y = valid 

	model = BiLSTMQA(12448, 100, 2)
	model.fit(train_x, train_y, vx=valid_x, vy=valid_y)


