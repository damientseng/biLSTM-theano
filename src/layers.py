import theano
import theano.tensor as tensor
import numpy as np
import timeit

import utils

class WordEmbeder(object):
	def __init__(self, voca_size, hidden_size):
		# word embedding matrix
		self.hidden_size = hidden_size
		self.Wemb = theano.shared(
			name="word embedding matrix",
			value = utils.init_norm(voca_size, hidden_size).astype(theano.config.floatX),
			borrow=True
			)
		self.params = [self.Wemb]

	def embed_it(self, inputs):
		"""
		type inputs: theano.tensor.matrix
		para inputs: A batch of sentences. Symbolic tensor inputs of shape (sentence_size, batch_size), elements are word indexes...
		"""
		sentence_size, batch_size = inputs.shape
		inputs = tensor.cast(inputs, "int32")
		outputs = self.Wemb[inputs.flatten()].reshape((sentence_size, batch_size, self.hidden_size))
		#shape( sentence_size, batch_size, word_emb_size)
		return outputs


class LSTM(object):
	def __init__(self, hidden_size):

		self.hidden_size = hidden_size

		# lstm W matrixes, Wf, Wi, Wo, Wc.
		self.W = theano.shared(name="W", value=utils.init_norm(self.hidden_size, 4*self.hidden_size), borrow=True)
		# lstm U matrixes, Uf, Ui, Uo, Uc.
		self.U = theano.shared(name="U", value=utils.init_norm(self.hidden_size, 4*self.hidden_size), borrow=True)
		# lstm b vectors, bf, bi, bo, bc.
		self.b = theano.shared(name="b", value=np.zeros( 4*self.hidden_size, dtype=theano.config.floatX ), borrow=True)

		self.params = [self.W, self.U, self.b]


	def forward(self, inputs, mask, h0=None, C0=None):
		"""
		param inputs: #(time_steps, batch_size, hidden_size).
		inputs: state_below
		"""
		if inputs.ndim == 3:
			batch_size = inputs.shape[1]
		else:
			batch_size = 1

		if h0 == None:
			h0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
		if C0 == None:
			C0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)

		def _step( m, X,   h_, C_,   W, U, b):
			XW = tensor.dot(X, W)    #(batch_size, 4*hidden_size)
			h_U = tensor.dot(h_, U)  #(batch_size, 4*hidden_size)
			#before activation,      #(batch_size, 4*hidden_size)
			bfr_actv = XW + h_U + b
			
			f = tensor.nnet.sigmoid( bfr_actv[:, 0:self.hidden_size] )                     #forget gate (batch_size, hidden_size)
			i = tensor.nnet.sigmoid( bfr_actv[:, 1*self.hidden_size:2*self.hidden_size] )   #input gate (batch_size, hidden_size)
			o = tensor.nnet.sigmoid( bfr_actv[:, 2*self.hidden_size:3*self.hidden_size] ) #output  gate (batch_size, hidden_size)
			Cp = tensor.tanh( bfr_actv[:, 3*self.hidden_size:4*self.hidden_size] )        #candi states (batch_size, hidden_size)

			C = i*Cp + f*C_
			C = m[:, None]*C + (1.0 - m)[:, None]*C_

			h = o*tensor.tanh( C ) 
			h = m[:, None]*h + (1.0 - m)[:, None]*h_

			h, C = tensor.cast(h, theano.config.floatX), tensor.cast(h, theano.config.floatX)
			return h, C

		outputs, updates = theano.scan(
			fn = _step,
			sequences = [mask, inputs],
			outputs_info = [h0, C0],
			non_sequences = [self.W, self.U, self.b]
			)

		hs, Cs = outputs
		return hs


class BiLSTM(object):
	def __init__(self, hidden_size, learning_rate=0.1):
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		self.params = []
		self._train = None
		self._predict = None

		self.fwd_lstm = LSTM(self.hidden_size)
		self.bwd_lstm = LSTM(self.hidden_size)
		self.params += self.fwd_lstm.params
		self.params += self.bwd_lstm.params

		self.Wfwd = theano.shared(name="Wfwd", value=utils.init_norm(self.hidden_size, self.hidden_size), borrow=True)
		self.Wbwd = theano.shared(name="Wbwd", value=utils.init_norm(self.hidden_size, self.hidden_size), borrow=True)
		self.bc = theano.shared(name="bc", value=np.zeros(self.hidden_size), borrow=True)

		self.params += [self.Wfwd, self.Wbwd, self.bc]

	def forward(self, inputs1, mask):
		rev_inputs = self.__reverse(inputs1, mask)
		hfs = self.fwd_lstm.forward(inputs1, mask)
		hbs = self.bwd_lstm.forward(rev_inputs, mask)
		hs = tensor.dot(hfs, self.Wfwd) + tensor.dot(hbs, self.Wbwd) + self.bc
		return tensor.cast(hs, theano.config.floatX)

	def __reverse(self, inputs, mask):
		source = inputs.dimshuffle(1, 0, 2)
		#aim = tensor.zeros_like(inputs) #(b, s, h)
		sz = mask.sum(axis=0)

		def _step(psource, psz):
			return tensor.concatenate([psource[psz-1::-1, :], tensor.zeros((mask.shape[0]-psz, self.hidden_size))] )

		outputs, updates = theano.scan(
			fn=_step,
			sequences=[source, sz]
			)
		return tensor.cast(outputs.dimshuffle(1, 0, 2), theano.config.floatX)
		#return outputs.dimshuffle(1, 0, 2) #shuffle back


class StackedBiLSTM(object):
	def __init__(self, voca_size, hidden_size, ydim, num_layers=2, learning_rate=0.1):
		self.hidden_size = hidden_size
		self.n_out = ydim
		self.learning_rate = learning_rate
		self.num_layers = num_layers
		self.layers = []
		self.params = []

		self.emb = WordEmbeder(voca_size, hidden_size)
		self.params += self.emb.params

		x = tensor.imatrix() #symbolic
		mask = tensor.imatrix()
		y = tensor.ivector()

		state_below = self.emb.embed_it(x)
		for _ in range(self.num_layers):
			binet = BiLSTM(self.hidden_size, self.learning_rate)
			self.layers += binet,
			self.params += binet.params
			state_below = binet.forward(state_below, mask)

		self.U = theano.shared(name="biU", value=utils.init_norm(self.hidden_size, self.n_out), borrow=True)
		self.by = theano.shared(name="by", value=np.zeros(self.n_out), borrow=True)
		self.params += [self.U, self.by]

		#mean pooling
		hs = state_below
		mp = (hs*mask[:,:,None]).sum(axis=0)
		mp = mp / mask.sum(axis=0)[:,None]

		#classifier
		pred_p = tensor.nnet.softmax(tensor.dot(mp, self.U) + self.by)
		pred_y = pred_p.argmax(axis=1)

		#nll
		off_set = 1e-8
		cost = -tensor.log( pred_p[tensor.arange(mask.shape[1]), y] + off_set ).mean()
		gparams = [tensor.grad(cost, param) for param in self.params]
		updates = [(param, param - self.learning_rate*gparam) for param, gparam in zip(self.params, gparams)]

		vinputs = tensor.imatrix("vinputs")#variable
		vmask = tensor.imatrix("vmask")
		vy = tensor.ivector("vy")
		
		self._train = theano.function(
			inputs=[vinputs, vmask, vy],
			outputs=cost,
			updates=updates,
			givens={x:vinputs, mask:vmask, y:vy}
			)

		self._predict = theano.function(
			inputs=[vinputs, vmask],
			outputs=pred_y,
			givens={x:vinputs, mask:vmask}
			)

	def train(self, inputs, mask, y):
		return self._train(inputs, mask, y)

	def predict(self, inputs, mask):
		return self._predict(inputs, mask)










