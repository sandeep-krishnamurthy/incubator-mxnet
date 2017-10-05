
Gluon Data API
**************


Overview
========

This document lists the data APIs in Gluon:

+-----------------------------+--------------------------------------------------------------------------------------------+
| ``mxnet.gluon.data``        | Dataset utilities.                                                                         |
+-----------------------------+--------------------------------------------------------------------------------------------+
| ``mxnet.gluon.data.vision`` | Dataset container.                                                                         |
+-----------------------------+--------------------------------------------------------------------------------------------+

The ``Gluon Data`` API, defined in the ``gluon.data`` package,
provides useful dataset loading and processing tools, as well as
common public datasets.

Warning: This package contains experimental APIs and may change in the near
  future.

In the rest of this document, we list routines provided by the
``gluon.data`` package.


Data
====

+-----------------------+--------------------------------------------------------------------------------------------+
| ``Dataset``           | Abstract dataset class.                                                                    |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``ArrayDataset``      | A dataset with a data array and a label array.                                             |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``RecordFileDataset`` | A dataset wrapping over a RecordIO (.rec) file.                                            |
+-----------------------+--------------------------------------------------------------------------------------------+

+-----------------------+--------------------------------------------------------------------------------------------+
| ``Sampler``           | Base class for samplers.                                                                   |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``SequentialSampler`` | Samples elements from [0, length) sequentially.                                            |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``RandomSampler``     | Samples elements from [0, length) randomly without replacement.                            |
+-----------------------+--------------------------------------------------------------------------------------------+
| ``BatchSampler``      | Wraps over another *Sampler* and return mini-batches of samples.                           |
+-----------------------+--------------------------------------------------------------------------------------------+

+----------------+--------------------------------------------------------------------------------------------+
| ``DataLoader`` | Loads data from a dataset and returns mini-batches of data.                                |
+----------------+--------------------------------------------------------------------------------------------+


Vision
------

+------------------------+--------------------------------------------------------------------------------------------+
| ``MNIST``              | MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist                     |
+------------------------+--------------------------------------------------------------------------------------------+
| ``FashionMNIST``       | A dataset of Zalando's article images consisting of fashion products,                      |
+------------------------+--------------------------------------------------------------------------------------------+
| ``CIFAR10``            | CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html      |
+------------------------+--------------------------------------------------------------------------------------------+
| ``CIFAR100``           | CIFAR100 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html     |
+------------------------+--------------------------------------------------------------------------------------------+
| ``ImageRecordDataset`` | A dataset wrapping over a RecordIO file containing images.                                 |
+------------------------+--------------------------------------------------------------------------------------------+
| ``ImageFolderDataset`` | A dataset for loading image files stored in a folder structure like:                       |
+------------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

Dataset utilities.

**class mxnet.gluon.data.ArrayDataset(data, label)**

   A dataset with a data array and a label array.

   The i-th sample is *(data[i], lable[i])*.

   :Parameters:
      * **data** (*array-like object*) -- The data array. Can be mxnet
        or numpy array.

      * **label** (*array-like object*) -- The label array. Can be
        mxnet or numpy array.

**class mxnet.gluon.data.BatchSampler(sampler, batch_size,
last_batch='keep')**

   Wraps over another *Sampler* and return mini-batches of samples.

   :Parameters:
      * **sampler** (*Sampler*) -- The source Sampler.

      * **batch_size** (*int*) -- Size of mini-batch.

      * **last_batch** (*{'keep'**, **'discard'**, **'rollover'}*) --

        Specifies how the last batch is handled if batch_size does not
        evenly divide sequence length.

        If 'keep', the last batch will be returned directly, but will
        contain less element than *batch_size* requires.

        If 'discard', the last batch will be discarded.

        If 'rollover', the remaining elements will be rolled over to
        the next iteration.

   -[ Examples ]-

   >>> sampler = gluon.data.SequentialSampler(10)
   >>> batch_sampler = gluon.data.BatchSampler(sampler, 3, 'keep')
   >>> list(batch_sampler)
   [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

**class mxnet.gluon.data.DataLoader(dataset, batch_size=None,
shuffle=False, sampler=None, last_batch=None, batch_sampler=None)**

   Loads data from a dataset and returns mini-batches of data.

   :Parameters:
      * **dataset** (*Dataset*) -- Source dataset. Note that numpy and
        mxnet arrays can be directly used as a Dataset.

      * **batch_size** (*int*) -- Size of mini-batch.

      * **shuffle** (*bool*) -- Whether to shuffle the samples.

      * **sampler** (*Sampler*) -- The sampler to use. Either specify
        sampler or shuffle, not both.

      * **last_batch** (*{'keep'**, **'discard'**, **'rollover'}*) --

        How to handle the last batch if batch_size does not evenly
        divide *len(dataset)*.

        keep - A batch with less samples than previous batches is
        returned. discard - The last batch is discarded if its
        incomplete. rollover - The remaining samples are rolled over
        to the next epoch.

      * **batch_sampler** (*Sampler*) -- A sampler that returns
        mini-batches. Do not specify batch_size, shuffle, sampler, and
        last_batch if batch_sampler is specified.

**class mxnet.gluon.data.Dataset**

   Abstract dataset class. All datasets should have this interface.

   Subclasses need to override *__getitem__*, which returns the i-th
   element, and *__len__*, which returns the total number elements.

   Note: An mxnet or numpy array can be directly used as a dataset.

**class mxnet.gluon.data.RandomSampler(length)**

   Samples elements from [0, length) randomly without replacement.

   :Parameters:
      **length** (*int*) -- Length of the sequence.

**class mxnet.gluon.data.RecordFileDataset(filename)**

   A dataset wrapping over a RecordIO (.rec) file.

   Each sample is a string representing the raw content of an record.

   :Parameters:
      **filename** (*str*) -- Path to rec file.

**class mxnet.gluon.data.Sampler**

   Base class for samplers.

   All samplers should subclass *Sampler* and define *__iter__* and
   *__len__* methods.

**class mxnet.gluon.data.SequentialSampler(length)**

   Samples elements from [0, length) sequentially.

   :Parameters:
      **length** (*int*) -- Length of the sequence.

Dataset container.

**class mxnet.gluon.data.vision.MNIST(root='~/.mxnet/datasets/mnist',
train=True, transform=None)**

   MNIST handwritten digits dataset from
   http://yann.lecun.com/exdb/mnist

   Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

   :Parameters:
      * **root** (*str**, **default '~/.mxnet/datasets/mnist'*) --
        Path to temp folder for storing data.

      * **train** (*bool**, **default True*) -- Whether to load the
        training or testing set.

      * **transform** (*function**, **default None*) -- A user defined
        callback that transforms each sample. For example:

   :param ::: transform=lambda data, label:
   (data.astype(np.float32)/255, label)

**class
mxnet.gluon.data.vision.FashionMNIST(root='~/.mxnet/datasets/fashion-mnist',
train=True, transform=None)**

   A dataset of Zalando's article images consisting of fashion
   products, a drop-in replacement of the original MNIST dataset from
   https://github.com/zalandoresearch/fashion-mnist

   Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

   :Parameters:
      * **root** (*str**, **default
        '~/.mxnet/datasets/fashion-mnist'*) -- Path to temp folder for
        storing data.

      * **train** (*bool**, **default True*) -- Whether to load the
        training or testing set.

      * **transform** (*function**, **default None*) -- A user defined
        callback that transforms each sample. For example:

   :param ::: transform=lambda data, label:
   (data.astype(np.float32)/255, label)

**class
mxnet.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10',
train=True, transform=None)**

   CIFAR10 image classification dataset from
   https://www.cs.toronto.edu/~kriz/cifar.html

   Each sample is an image (in 3D NDArray) with shape (32, 32, 1).

   :Parameters:
      * **root** (*str**, **default '~/.mxnet/datasets/cifar10'*) --
        Path to temp folder for storing data.

      * **train** (*bool**, **default True*) -- Whether to load the
        training or testing set.

      * **transform** (*function**, **default None*) -- A user defined
        callback that transforms each sample. For example:

   :param ::: transform=lambda data, label:
   (data.astype(np.float32)/255, label)

**class
mxnet.gluon.data.vision.CIFAR100(root='~/.mxnet/datasets/cifar100',
fine_label=False, train=True, transform=None)**

   CIFAR100 image classification dataset from
   https://www.cs.toronto.edu/~kriz/cifar.html

   Each sample is an image (in 3D NDArray) with shape (32, 32, 1).

   :Parameters:
      * **root** (*str**, **default '~/.mxnet/datasets/cifar100'*) --
        Path to temp folder for storing data.

      * **fine_label** (*bool**, **default False*) -- Whether to load
        the fine-grained (100 classes) or coarse-grained (20
        super-classes) labels.

      * **train** (*bool**, **default True*) -- Whether to load the
        training or testing set.

      * **transform** (*function**, **default None*) -- A user defined
        callback that transforms each sample. For example:

   :param ::: transform=lambda data, label:
   (data.astype(np.float32)/255, label)

**class mxnet.gluon.data.vision.ImageRecordDataset(filename, flag=1,
transform=None)**

   A dataset wrapping over a RecordIO file containing images.

   Each sample is an image and its corresponding label.

   :Parameters:
      * **filename** (*str*) -- Path to rec file.

      * **flag** (*{0**, **1}**, **default 1*) --

        If 0, always convert images to greyscale.

        If 1, always convert images to colored (RGB).

      * **transform** (*function**, **default None*) -- A user defined
        callback that transforms each sample. For example:

   :param ::: transform=lambda data, label:
   (data.astype(np.float32)/255, label)

**class mxnet.gluon.data.vision.ImageFolderDataset(root, flag=1,
transform=None)**

   A dataset for loading image files stored in a folder structure
   like:

   ::

      root/car/0001.jpg
      root/car/xxxa.jpg
      root/car/yyyb.jpg
      root/bus/123.jpg
      root/bus/023.jpg
      root/bus/wwww.jpg

   :Parameters:
      * **root** (*str*) -- Path to root directory.

      * **flag** (*{0**, **1}**, **default 1*) -- If 0, always convert
        loaded images to greyscale (1 channel). If 1, always convert
        loaded images to colored (3 channels).

      * **transform** (*callable**, **default None*) -- A function
        that takes data and label and transforms them:

   :param ::: transform = lambda data, label:
   (data.astype(np.float32)/255, label)

   ``synsets``

      *list* -- List of class names. *synsets[i]* is the name for the
      integer label *i*

   ``items``

      *list of tuples* -- List of all images in (filename, label)
      pairs.
