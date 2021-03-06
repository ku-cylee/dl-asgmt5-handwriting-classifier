import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        # W: (cout, cin, wfil, hfil)
        # b: (1, cout, 1, 1)
        # x: (b, cin, win, hin)
        # y: (b, cout, wout, hout)

        b, cin, _, _ = x.shape
        cout, _, wfil, hfil = self.W.shape

        reshaped_W = self.W.reshape(cout, 1, -1)

        windows = np.squeeze(view_as_windows(x, (1, cin, wfil, hfil)), axis=(1, 4))
        _, wout, hout, _, _, _ = windows.shape
        reshaped_windows = windows.reshape(b, wout, hout, -1, 1)

        return np.swapaxes(np.squeeze(reshaped_W.dot(reshaped_windows), axis=(1, -1)), 0, 1) + self.b

    def backprop(self, x, dLdy):
        cout, cin, wfil, hfil = self.W.shape

        # dLdW: (cout, cin, wfil, hfil)
        x_windows = np.squeeze(view_as_windows(x, (1, 1, wfil, hfil)), axis=(4, 5)) \
                      .transpose(1, 4, 5, 0, 2, 3) \
                      .reshape((cin, wfil, hfil, -1, 1))
        dLdy_flattened = np.swapaxes(dLdy, 0, 1).reshape(cout, 1, -1)
        dLdW = np.squeeze(dLdy_flattened.dot(x_windows), axis=(1, -1))

        # dLdx: (b, cin, win, hin)
        dLdy_padded = np.pad(dLdy, ((0, 0), (0, 0), (wfil - 1, wfil - 1), (hfil - 1, hfil - 1)), mode='constant')
        dLdy_windows = np.squeeze(view_as_windows(dLdy_padded, (1, cout, wfil, hfil)), axis=(1, 4))
        dLdy_flattened = dLdy_windows.reshape(dLdy_windows.shape[:3] + (1, -1))
        W_flipped = np.swapaxes(np.flip(self.W, axis=(2, 3)), 0, 1).reshape(cin, -1, 1)
        dLdx = np.squeeze(dLdy_flattened.dot(W_flipped), axis=(3, 5)).transpose(0, 3, 1, 2)

        # dLdb: (1, cout, 1, 1)
        dLdb = dLdy.sum(axis=3).sum(axis=2).sum(axis=0).reshape(self.b.shape)

        return dLdx, dLdW, dLdb

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        # x: (b, cin, win, hin)
        # y: (b, cout=cin, wout, wout)
        windows = view_as_windows(x,
                                  (1, 1, self.pool_size, self.pool_size),
                                  step=(1, 1, self.stride, self.stride))
        return np.max(windows, axis=(4, 5, 6, 7))

    def backprop(self, x, dLdy):
        # x: (b, cin, win, hin)
        # y: (b, cout=cin, wout, wout)
        # dLdy: (b, cout=cin, wout, wout)
        windows = view_as_windows(x,
                                  (1, 1, self.pool_size, self.pool_size),
                                  step=(1, 1, self.stride, self.stride))
        windowed_x = np.squeeze(windows, axis=(4, 5))

        y = self.forward(x)
        y_extended = y[:, :, :, :, None, None]
        _, _, wout, hout = y.shape

        dLdx = np.zeros_like(x, dtype=np.float64)
        for wout_idx in range(wout):
            for hout_idx in range(hout):
                window = windowed_x[:, :, wout_idx, hout_idx]
                max_value = y_extended[:, :, wout_idx, hout_idx]
                dLdy_value = dLdy[:, :, wout_idx, hout_idx, None, None]
                dLdx_submatrix = np.where(window == max_value, dLdy_value, 0)

                win_idx = wout_idx * self.stride
                hin_idx = hout_idx * self.stride
                dLdx[:, :, win_idx:win_idx + self.pool_size, hin_idx:hin_idx + self.pool_size] += dLdx_submatrix

        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        return x.reshape(x.shape[0], -1) @ self.W.T + self.b.T

    def backprop(self,x,dLdy):
        dLdW = dLdy.T @ x.reshape(x.shape[0], -1)
        dLdb = dLdy.sum(axis=0)
        dLdx = (dLdy @ self.W).reshape(x.shape)
        return dLdx, dLdW, dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.where(x > 0, x, 0)
    
    def backprop(self, x, dLdy):
        return np.where(x > 0, dLdy, 0)


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        exp_scores = np.exp(x)
        exp_scores_sum = exp_scores.sum(axis=1).reshape(-1, 1)
        return exp_scores / exp_scores_sum

    def backprop(self, x, dLdy):
        y = self.forward(x)
        y_prods = y[:, :, None] @ y[:, None, :]
        return dLdy * y - np.squeeze(dLdy[:, None, :] @ y_prods, axis=1)

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        batch_size, _ = x.shape
        real_scores = x[np.arange(batch_size), y]
        return - np.average(np.log(real_scores))

    def backprop(self, x, y):
        batch_size, _ = x.shape
        bch_size_neginv = - 1 / batch_size

        y_mtx = np.zeros_like(x)
        for b_idx in range(batch_size):
            y_mtx[b_idx, y[b_idx]] = bch_size_neginv

        return y_mtx / x
