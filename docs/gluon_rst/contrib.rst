
Gluon Contrib API
*****************


Overview
========

This document lists the contrib APIs in Gluon:

+-------------------------+--------------------------------------------------------------------------------------------+
| ``mxnet.gluon.contrib`` | Contrib neural network module.                                                             |
+-------------------------+--------------------------------------------------------------------------------------------+

The ``Gluon Contrib`` API, defined in the ``gluon.contrib`` package,
provides many useful experimental APIs for new features. This is a
place for the community to try out the new features, so that feature
contributors can receive feedback.

Warning: This package contains experimental APIs and may change in the near
  future.

In the rest of this document, we list routines provided by the
``gluon.contrib`` package.


Contrib
=======

+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.VariationalDropoutCell`` | Applies Variational Dropout on base cell.                                                  |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv1DRNNCell``          | 1D Convolutional RNN cell.                                                                 |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv2DRNNCell``          | 2D Convolutional RNN cell.                                                                 |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv3DRNNCell``          | 3D Convolutional RNN cells                                                                 |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv1DLSTMCell``         | 1D Convolutional LSTM network cell.                                                        |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv2DLSTMCell``         | 2D Convolutional LSTM network cell.                                                        |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv3DLSTMCell``         | 3D Convolutional LSTM network cell.                                                        |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv1DGRUCell``          | 1D Convolutional Gated Rectified Unit (GRU) network cell.                                  |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv2DGRUCell``          | 2D Convolutional Gated Rectified Unit (GRU) network cell.                                  |
+--------------------------------+--------------------------------------------------------------------------------------------+
| ``rnn.Conv3DGRUCell``          | 3D Convolutional Gated Rectified Unit (GRU) network cell.                                  |
+--------------------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Contrib neural network module.

Contrib recurrent neural network module.

**class mxnet.gluon.contrib.rnn.Conv1DRNNCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, ), i2h_dilate=(1,
), h2h_dilate=(1, ), i2h_weight_initializer=None,
h2h_weight_initializer=None, i2h_bias_initializer='zeros',
h2h_bias_initializer='zeros', conv_layout='NCW', activation='tanh',
prefix=None, params=None)**

   1D Convolutional RNN cell.

      h_t = tanh(W_i \ast x_t + R_i \ast h_{t-1} + b_i)

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCW' the shape should
        be (C, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default**
        (**0**,****)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**,****)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**,****)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCW' and
        'NWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function. If argument type is string, it's
        equivalent to nn.Activation(act_type=str). See `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_rnn_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv2DRNNCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0),
i2h_dilate=(1, 1), h2h_dilate=(1, 1), i2h_weight_initializer=None,
h2h_weight_initializer=None, i2h_bias_initializer='zeros',
h2h_bias_initializer='zeros', conv_layout='NCHW', activation='tanh',
prefix=None, params=None)**

   2D Convolutional RNN cell.

      h_t = tanh(W_i \ast x_t + R_i \ast h_{t-1} + b_i)

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCHW' the shape should
        be (C, H, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default** (**0**,
        **0**)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCHW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCHW'
        and 'NHWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function. If argument type is string, it's
        equivalent to nn.Activation(act_type=str). See `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_rnn_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv3DRNNCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0, 0),
i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
conv_layout='NCDHW', activation='tanh', prefix=None, params=None)**

   3D Convolutional RNN cells

      h_t = tanh(W_i \ast x_t + R_i \ast h_{t-1} + b_i)

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCDHW' the shape
        should be (C, D, H, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default** (**0**,
        **0**, **0**)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**, **1**)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**, **1**)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCDHW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCDHW'
        and 'NDHWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function. If argument type is string, it's
        equivalent to nn.Activation(act_type=str). See `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_rnn_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv1DLSTMCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, ), i2h_dilate=(1,
), h2h_dilate=(1, ), i2h_weight_initializer=None,
h2h_weight_initializer=None, i2h_bias_initializer='zeros',
h2h_bias_initializer='zeros', conv_layout='NCW', activation='tanh',
prefix=None, params=None)**

   1D Convolutional LSTM network cell.

   "Convolutional LSTM Network: A Machine Learning Approach for
   Precipitation Nowcasting" paper. Xingjian et al. NIPS2015

      \begin{array}{ll} i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} +
      b_i) \\ f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
      o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
      c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\ c_t
      = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\ h_t = o_t \circ
      tanh(c_t) \\ \end{array}

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCW' the shape should
        be (C, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default**
        (**0**,****)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**,****)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**,****)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCW' and
        'NWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function used in c^prime_t. If argument
        type is string, it's equivalent to
        nn.Activation(act_type=str). See `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_lstm_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv2DLSTMCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0),
i2h_dilate=(1, 1), h2h_dilate=(1, 1), i2h_weight_initializer=None,
h2h_weight_initializer=None, i2h_bias_initializer='zeros',
h2h_bias_initializer='zeros', conv_layout='NCHW', activation='tanh',
prefix=None, params=None)**

   2D Convolutional LSTM network cell.

   "Convolutional LSTM Network: A Machine Learning Approach for
   Precipitation Nowcasting" paper. Xingjian et al. NIPS2015

      \begin{array}{ll} i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} +
      b_i) \\ f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
      o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
      c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\ c_t
      = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\ h_t = o_t \circ
      tanh(c_t) \\ \end{array}

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCHW' the shape should
        be (C, H, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default** (**0**,
        **0**)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCHW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCHW'
        and 'NHWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function used in c^prime_t. If argument
        type is string, it's equivalent to
        nn.Activation(act_type=str). See `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_lstm_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv3DLSTMCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0, 0),
i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
conv_layout='NCDHW', activation='tanh', prefix=None, params=None)**

   3D Convolutional LSTM network cell.

   "Convolutional LSTM Network: A Machine Learning Approach for
   Precipitation Nowcasting" paper. Xingjian et al. NIPS2015

      \begin{array}{ll} i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} +
      b_i) \\ f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
      o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
      c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\ c_t
      = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\ h_t = o_t \circ
      tanh(c_t) \\ \end{array}

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCDHW' the shape
        should be (C, D, H, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default** (**0**,
        **0**, **0**)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**, **1**)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**, **1**)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCDHW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCDHW'
        and 'NDHWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function used in c^prime_t. If argument
        type is string, it's equivalent to
        nn.Activation(act_type=str). See `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_lstm_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv1DGRUCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, ), i2h_dilate=(1,
), h2h_dilate=(1, ), i2h_weight_initializer=None,
h2h_weight_initializer=None, i2h_bias_initializer='zeros',
h2h_bias_initializer='zeros', conv_layout='NCW', activation='tanh',
prefix=None, params=None)**

   1D Convolutional Gated Rectified Unit (GRU) network cell.

      \begin{array}{ll} r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} +
      b_r) \\ z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
      n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} +
      b_n)) \\ h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
      \end{array}

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCW' the shape should
        be (C, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default**
        (**0**,****)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**,****)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**,****)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCW' and
        'NWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function used in n_t. If argument type is
        string, it's equivalent to nn.Activation(act_type=str). See
        `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_gru_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv2DGRUCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0),
i2h_dilate=(1, 1), h2h_dilate=(1, 1), i2h_weight_initializer=None,
h2h_weight_initializer=None, i2h_bias_initializer='zeros',
h2h_bias_initializer='zeros', conv_layout='NCHW', activation='tanh',
prefix=None, params=None)**

   2D Convolutional Gated Rectified Unit (GRU) network cell.

      \begin{array}{ll} r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} +
      b_r) \\ z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
      n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} +
      b_n)) \\ h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
      \end{array}

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCHW' the shape should
        be (C, H, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default** (**0**,
        **0**)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCHW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCHW'
        and 'NHWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function used in n_t. If argument type is
        string, it's equivalent to nn.Activation(act_type=str). See
        `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_gru_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.Conv3DGRUCell(input_shape,
hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0, 0),
i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
i2h_weight_initializer=None, h2h_weight_initializer=None,
i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
conv_layout='NCDHW', activation='tanh', prefix=None, params=None)**

   3D Convolutional Gated Rectified Unit (GRU) network cell.

      \begin{array}{ll} r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} +
      b_r) \\ z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
      n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} +
      b_n)) \\ h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
      \end{array}

   :Parameters:
      * **input_shape** (*tuple of int*) -- Input tensor shape at each
        time step for each sample, excluding dimension of the batch
        size and sequence length. Must be consistent with
        *conv_layout*. For example, for layout 'NCDHW' the shape
        should be (C, D, H, W).

      * **hidden_channels** (*int*) -- Number of output channels.

      * **i2h_kernel** (*int** or **tuple of int*) -- Input
        convolution kernel sizes.

      * **h2h_kernel** (*int** or **tuple of int*) -- Recurrent
        convolution kernel sizes. Only odd-numbered sizes are
        supported.

      * **i2h_pad** (*int** or **tuple of int**, **default** (**0**,
        **0**, **0**)***) -- Pad for input convolution.

      * **i2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**, **1**)***) -- Input convolution dilate.

      * **h2h_dilate** (*int** or **tuple of int**, **default**
        (**1**, **1**, **1**)***) -- Recurrent convolution dilate.

      * **i2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the input weights matrix, used for the
        input convolutions.

      * **h2h_weight_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_)
        -- Initializer for the recurrent weights matrix, used for the
        input convolutions.

      * **i2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the input convolution
        bias vectors.

      * **h2h_bias_initializer** (*str** or *`Initializer
        <../optimization/optimization.rst#mxnet.initializer.Initializer>`_*,
        **default zeros*) -- Initializer for the recurrent convolution
        bias vectors.

      * **conv_layout** (*str**, **default 'NCDHW'*) -- Layout for all
        convolution inputs, outputs and weights. Options are 'NCDHW'
        and 'NDHWC'.

      * **activation** (*str** or **Block**, **default 'tanh'*) --
        Type of activation function used in n_t. If argument type is
        string, it's equivalent to nn.Activation(act_type=str). See
        `Activation()
        <../ndarray/ndarray.rst#mxnet.ndarray.Activation>`_ for
        available choices. Alternatively, other activation blocks such
        as nn.LeakyReLU can be used.

      * **prefix** (str, default '>>conv_gru_<<') -- Prefix for name
        of layers (and name of weight if params is None).

      * **params** (`RNNParams
        <../symbol/rnn.rst#mxnet.rnn.RNNParams>`_*, **default None*)
        -- Container for weight sharing between cells. Created if
        None.

**class mxnet.gluon.contrib.rnn.VariationalDropoutCell(base_cell,
drop_inputs=0.0, drop_states=0.0, drop_outputs=0.0)**

   Applies Variational Dropout on base cell.
   (https://arxiv.org/pdf/1512.05287.pdf,

   ..

      https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf).

   Variational dropout uses the same dropout mask across time-steps.
   It can be applied to RNN inputs, outputs, and states. The masks for
   them are not shared.

   The dropout mask is initialized when stepping forward for the first
   time and will remain the same until .reset() is called. Thus, if
   using the cell and stepping manually without calling .unroll(), the
   .reset() should be called after each sequence.

   :Parameters:
      * **base_cell** (`RecurrentCell
        <rnn.rst#mxnet.gluon.rnn.RecurrentCell>`_) -- The cell on
        which to perform variational dropout.

      * **drop_inputs** (*float**, **default 0.*) -- The dropout rate
        for inputs. Won't apply dropout if it equals 0.

      * **drop_states** (*float**, **default 0.*) -- The dropout rate
        for state inputs on the first state channel. Won't apply
        dropout if it equals 0.

      * **drop_outputs** (*float**, **default 0.*) -- The dropout rate
        for outputs. Won't apply dropout if it equals 0.

   **unroll(length, inputs, begin_state=None, layout='NTC',
   merge_outputs=None)**

      Unrolls an RNN cell across time steps.

      :Parameters:
         * **length** (*int*) -- Number of steps to unroll.

         * **inputs** (`Symbol
           <../symbol/symbol.rst#mxnet.symbol.Symbol>`_*, **list of
           Symbol**, or **None*) --

           If *inputs* is a single Symbol (usually the output of
           Embedding symbol), it should have shape (batch_size,
           length, ...) if *layout* is 'NTC', or (length, batch_size,
           ...) if *layout* is 'TNC'.

           If *inputs* is a list of symbols (usually output of
           previous unroll), they should all have shape (batch_size,
           ...).

         * **begin_state** (*nested list of Symbol**, **optional*) --
           Input states created by *begin_state()* or output state of
           another cell. Created from *begin_state()* if *None*.

         * **layout** (*str**, **optional*) -- *layout* of input
           symbol. Only used if inputs is a single Symbol.

         * **merge_outputs** (*bool**, **optional*) -- If *False*,
           returns outputs as a list of Symbols. If *True*,
           concatenates output across time steps and returns a single
           symbol with shape (batch_size, length, ...) if layout is
           'NTC', or (length, batch_size, ...) if layout is 'TNC'. If
           *None*, output whatever is faster.

      :Returns:
         * **outputs** (*list of Symbol or Symbol*) -- Symbol (if
           *merge_outputs* is True) or list of Symbols (if
           *merge_outputs* is False) corresponding to the output from
           the RNN from this unrolling.

         * **states** (*list of Symbol*) -- The new state of this RNN
           after this unrolling. The type of this symbol is same as
           the output of *begin_state()*.
