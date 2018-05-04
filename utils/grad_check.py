import numpy as np

def __compute_approx_grads(nn, x, y, eps=1e-4):
    approx_grads = []
    feed_forward = lambda inp: nn._NeuralNetwork__feedforward(inp, is_training=True)

    for layer in nn.layers:
        assert(layer.dropout_prob == 0.0), "O Gradient Checking não pode ser aplicado em redes com DROPOUT"

        w_ori = layer.weights.copy()
        w_ravel = w_ori.ravel()
        w_shape = w_ori.shape

        for i in range(w_ravel.size):
            w_plus = w_ravel.copy()
            w_plus[i] += eps
            layer.weights = w_plus.reshape(w_shape)
            J_plus = nn.cost_func(y, feed_forward(x)) + (1.0/y.shape[0])*layer.reg_strength*layer.reg_func(layer.weights)

            w_minus = w_ravel.copy()
            w_minus[i] -= eps
            layer.weights = w_minus.reshape(w_shape)
            J_minus = nn.cost_func(y, feed_forward(x)) + (1.0/y.shape[0])*layer.reg_strength*layer.reg_func(layer.weights)
            approx_grads.append((J_plus - J_minus) / (2.0*eps))
        layer.weights = w_ori

    return approx_grads

def gradient_checking(nn, x, y, eps=1e-4, verbose=False, verbose_precision=5):
    from copy import deepcopy
    nn_copy = deepcopy(nn)

    nn.fit(x, y, epochs=0)
    grads = np.concatenate([layer._dweights.ravel() for layer in nn.layers])

    approx_grads = __compute_approx_grads(nn_copy, x, y, eps)

    is_close = np.allclose(grads, approx_grads)
    print("{}".format("\033[92mGRADIENTS OK" if is_close else "\033[91mGRADIENTS FAIL"))

    norm_num = np.linalg.norm(grads - approx_grads)
    norm_den = np.linalg.norm(grads) + np.linalg.norm(approx_grads)
    error = norm_num / norm_den
    print("Relative error:", error)

    if verbose:
        np.set_printoptions(precision=verbose_precision, linewidth=200, suppress=True)
        print("Gradientes:", grads)
        print("Aproximado:", np.array(approx_grads))