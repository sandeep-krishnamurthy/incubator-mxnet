# Gluon Package


```eval_rst
.. currentmodule:: mxnet.gluon
```

```eval_rst
.. warning:: This package is currently experimental and may change in the near future.
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

## Overview

Gluon is a high-level interface for Deep Learning frameworks like MXNet
designed to be easy to use while keeping most of the flexibility of low level
API. Gluon supports both imperative and symbolic programming, making it easy to
train complex models imperatively in Python and then deploy with symbolic graph
in C++ and Scala.

```eval_rst
.. toctree::
   :maxdepth: 1

   autograd.md
   nn.md
   rnn.md
   loss.md
   data.md
   ndarray.md
   sparse.md
   model_zoo.md
   contrib.md
```

## Autograd

```eval_rst
.. currentmodule:: mxnet.autograd

.. autosummary::
    :nosignatures:

    record
    pause
    train_mode
    predict_mode
    backward
    set_training
    is_training
    set_recording
    is_recording
    mark_variables
    Function
```

## Containers

```eval_rst
.. currentmodule:: mxnet.gluon

.. autosummary::
    :nosignatures:

    Block
    HybridBlock
    SymbolBlock
```

## Data

```eval_rst
.. currentmodule:: mxnet.gluon.data
```

```eval_rst
.. autosummary::
    :nosignatures:

    Dataset
    ArrayDataset
    RecordFileDataset
```

```eval_rst
.. autosummary::
    :nosignatures:

    Sampler
    SequentialSampler
    RandomSampler
    BatchSampler
```

```eval_rst
.. autosummary::
    :nosignatures:

    DataLoader
```

### Vision

```eval_rst
.. currentmodule:: mxnet.gluon.data.vision
```

```eval_rst
.. autosummary::
    :nosignatures:

    MNIST
    FashionMNIST
    CIFAR10
    CIFAR100
    ImageRecordDataset
    ImageFolderDataset
```

## NDArray

```eval_rst
.. currentmodule:: mxnet.ndarray

.. autosummary::
    :nosignatures:

    mxnet.ndarray
```

### Sparse NDArray

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray
    sparse.CSRNDArray
    sparse.RowSparseNDArray
```

## Neural Network API

```eval_rst
.. currentmodule:: mxnet.gluon.nn

.. autosummary::
    :nosignatures:

    Sequential
    HybridSequential
```

## Parameter

```eval_rst
.. autosummary::
    :nosignatures:

    Parameter
    ParameterDict
```

## Recurrent Neural Network API

```eval_rst
.. currentmodule:: mxnet.gluon.rnn

.. autosummary::
    :nosignatures:

    RNNCell
    LSTMCell
    GRUCell
    RecurrentCell
    SequentialRNNCell
    BidirectionalCell
    DropoutCell
    ZoneoutCell
    ResidualCell
```

## Updater

```eval_rst
.. currentmodule:: mxnet.gluon

.. autosummary::
    :nosignatures:

    Trainer
```

## Utilities

```eval_rst
.. currentmodule:: mxnet.gluon.utils
```


```eval_rst
.. autosummary::
    :nosignatures:

    split_data
    split_and_load
    clip_global_norm
```
