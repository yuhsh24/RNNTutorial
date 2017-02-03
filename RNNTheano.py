import numpy as np
import theano.tensor as T
import theano
from utils import *
import operator

class RNNTheano:

    def __init__(self, word_dim, hidden_dim, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # Theano : create shared variables
        self.U = theano.shared(name="U", value=U.astype(theano.config.floatX))
        self.V = theano.shared(name="V", value=V.astype(theano.config.floatX))
        self.W = theano.shared(name="W", value=W.astype(theano.config.floatX))
        # Store the theano graph
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        # Build the theano graph
        U = self.U
        V = self.V
        W = self.W
        x = T.ivector("x")
        y = T.ivector("y")

        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:, x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            # Because the o_t is a two-dimesion array, so return the first element
            return o_t[0], s_t

        [o, s], updates = theano.scan(forward_prop_step,
                                      sequences=x,
                                      outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
                                      non_sequences=[U, V, W],
                                      truncate_gradient=self.bptt_truncate,
                                      strict=True)

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Gradient
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)

        # Assign function
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        # SGD
        learning_rate = T.dscalar("learning_rate")
        self.sgd_step = theano.function([x, y, learning_rate], [],
                                        updates=[(self.U, self.U - learning_rate * dU),
                                                 (self.V, self.V - learning_rate * dV),
                                                 (self.W, self.W - learning_rate * dW)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        sample_count = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(sample_count)

def gradient_check(model, x, y, h=0.001, error_threshold=0.01):
    # change the bptt_truncate
    model.bptt_truncate = 10000
    # calculate the gradient
    bptt_gradients = model.bptt(x, y)
    # List the parameter name
    model_parameters = ["U", "V", "W"]
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # get the parameter
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with the size %d." % (pname, np.prod(parameter.shape))
        # Iter the element
        it = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            ix = it.multi_index
            # Save origin value
            original_value = parameter[ix]
            # Estimate the gradient
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus) / (2 * h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated by back propagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate the relative error
            relative_error = np.abs(estimated_gradient - backprop_gradient) / (np.abs(estimated_gradient) + np.abs(backprop_gradient) + 1e-6)
            # if the error is too larage
            if relative_error > error_threshold:
                print "Gradient Check Error: parameter=%s, ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated gradient: %f" % estimated_gradient
                print "Backpropogation gradient: %f" % bptt_gradients
                print "Relative error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % pname