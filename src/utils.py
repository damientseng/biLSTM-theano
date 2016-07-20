import theano 
import numpy as np

def init_ortho(size):
	return np.concatenate(
			[ ortho_matrix( size ),
			  ortho_matrix( size ),
			  ortho_matrix( size ),
			  ortho_matrix( size ), ], axis=1).astype(theano.config.floatX)

def init_uniform(n_in, n_out):
	W = np.asarray(
		np.random.uniform(
			low=-4.*np.sqrt(6. / (n_in + n_out)),
			high=4.*np.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
			),
		dtype=theano.config.floatX
		)
	return W 

def init_norm(n_in, n_out):
	W = np.asarray(np.random.randn(n_in, n_out)*0.1, dtype=theano.config.floatX)
	return W