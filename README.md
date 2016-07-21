# biLSTM-theano
A theano implementation of biLSTM network.
The structure is illustrated in the following pic. A sequence is processed from two directions, which may capture more 
interesting information than an ordinary LSTM. Serveral biLSTMs can be stacked together to form a deep network.
This repository is specifically implemented to do text classification on the [movie review dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data).
Accuracy > 0.8 is easily obtained with two layers of biLSTM.
![](https://github.com/saltypaul/biLSTM-theano/blob/master/Pics/biLSTM.png)
