
Sparse NDArray API
******************


Overview
========

This document lists the routines of the *n*-dimensional sparse array
package:

+-------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `mxnet.ndarray.sparse                                 | Sparse NDArray API of MXNet.                                                               |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------+

The ``CSRNDArray`` and ``RowSparseNDArray`` API, defined in the
``ndarray.sparse`` package, provides imperative sparse tensor
operations on CPU.

An ``CSRNDArray`` inherits from ``NDArray``, and represents a
two-dimensional, fixed-size array in compressed sparse row format.

::

   >>> x = mx.nd.array([[1, 0], [0, 0], [2, 3]])
   >>> csr = x.tostype('csr')
   >>> type(csr)
   <class 'mxnet.ndarray.sparse.CSRNDArray'>
   >>> csr.shape
   (3, 2)
   >>> csr.data.asnumpy()
   array([ 1.  2.  3.], dtype=float32)
   >>> csr.indices.asnumpy()
   array([0, 0, 1])
   >>> csr.indptr.asnumpy()
   array([0, 1, 1, 3])
   >>> csr.stype
   'csr'

An ``RowSparseNDArray`` inherits from ``NDArray``, and represents a
multi-dimensional, fixed-size array in row sparse format.

::

   >>> x = mx.nd.array([[1, 0], [0, 0], [2, 3]])
   >>> row_sparse = x.tostype('row_sparse')
   >>> type(row_sparse)
   <class 'mxnet.ndarray.sparse.RowSparseNDArray'>
   >>> row_sparse.data.asnumpy()
   array([[ 1.  0.],
          [ 2.  3.]], dtype=float32)
   >>> row_sparse.indices.asnumpy()
   array([0, 2])
   >>> row_sparse.stype
   'row_sparse'

Note: ``mxnet.ndarray.sparse`` is similar to ``mxnet.ndarray`` in some
  aspects. But the differences are not negligible. For instance:

  * Only a subset of operators in ``mxnet.ndarray`` have specialized
    implementations in ``mxnet.ndarray.sparse``. Operators such as
    reduction and broadcasting do not have sparse implementations yet.

  * The storage types (``stype``) of sparse operators' outputs depend
    on the storage types of inputs. By default the operators not
    available in ``mxnet.ndarray.sparse`` infer "default" (dense)
    storage type for outputs. Please refer to the API reference
    section for further details on specific operators.

  * GPU support for ``mxnet.ndarray.sparse`` is experimental.

In the rest of this document, we first overview the methods provided
by the ``ndarray.sparse.CSRNDArray`` class and the
``ndarray.sparse.RowSparseNDArray`` class, and then list other
routines provided by the ``ndarray.sparse`` package.

The ``ndarray.sparse`` package provides several classes:

+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray                                                     | A sparse representation of 2D NDArray in the Compressed Sparse Row format.                 |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray                                               | A sparse representation of a set of NDArray row slices at given indices.                   |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+

We summarize the interface for each class in the following sections.


The ``CSRNDArray`` class
========================


Array attributes
----------------

+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.shape                                                 | Tuple of array dimensions.                                                                 |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.context                                               | Device context of the array.                                                               |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.dtype                                                 | Data-type of the array's elements.                                                         |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.stype                                                 | Storage-type of the array.                                                                 |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.data                                                  | A deep copy NDArray of the data array of the CSRNDArray.                                   |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.indices                                               | A deep copy NDArray of the indices array of the CSRNDArray.                                |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.indptr                                                | A deep copy NDArray of the indptr array of the CSRNDArray.                                 |
+-------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array conversion
----------------

+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.copy                                                        | Makes a copy of this ``NDArray``, keeping the same context.                                |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.copyto                                                      | Copies the value of this array to another array.                                           |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.as_in_context                                               | Returns an array on the target device with the same value as this array.                   |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.asnumpy                                                     | Return a dense ``numpy.ndarray`` object with value copied from this array                  |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.asscalar                                                    | Returns a scalar whose value is copied from this array.                                    |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.astype                                                      | Returns a copy of the array after casting to a specified type.                             |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.tostype                                                     | Return a copy of the array with chosen storage type.                                       |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array creation
--------------

+----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.zeros_like                                               | Convenience fluent method for `zeros_like()                                                |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray.zeros_like>`_ | <../ndarray/sparse.rst#mxnet.ndarray.sparse.zeros_like>`_.                                 |
+----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Indexing
--------

+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.__getitem__                                               | x.__getitem__(i) <=> x[i]                                                                  |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.__setitem__                                               | x.__setitem__(i, y) <=> x[i]=y                                                             |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.slice                                                     | Convenience fluent method for `slice()                                                     |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray.slice>`_       | <../ndarray/sparse.rst#mxnet.ndarray.sparse.slice>`_.                                      |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Lazy evaluation
---------------

+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `CSRNDArray.wait_to_read                                               | Waits until all previous write operations on the current array are finished.               |
+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


The ``RowSparseNDArray`` class
==============================


Array attributes
----------------

+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.shape                                                 | Tuple of array dimensions.                                                                 |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.context                                               | Device context of the array.                                                               |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.dtype                                                 | Data-type of the array's elements.                                                         |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.stype                                                 | Storage-type of the array.                                                                 |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.data                                                  | A deep copy NDArray of the data array of the RowSparseNDArray.                             |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.indices                                               | A deep copy NDArray of the indices array of the RowSparseNDArray.                          |
+-------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array conversion
----------------

+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.copy                                                        | Makes a copy of this ``NDArray``, keeping the same context.                                |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.copyto                                                      | Copies the value of this array to another array.                                           |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.as_in_context                                               | Returns an array on the target device with the same value as this array.                   |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.asnumpy                                                     | Return a dense ``numpy.ndarray`` object with value copied from this array                  |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.asscalar                                                    | Returns a scalar whose value is copied from this array.                                    |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.astype                                                      | Returns a copy of the array after casting to a specified type.                             |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.tostype                                                     | Return a copy of the array with chosen storage type.                                       |
+-------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array creation
--------------

+----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.zeros_like                                               | Convenience fluent method for `zeros_like()                                                |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray.zeros_like>`_ | <../ndarray/sparse.rst#mxnet.ndarray.sparse.zeros_like>`_.                                 |
+----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array rounding
--------------

+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.round                                               | Convenience fluent method for `round()                                                     |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray.round>`_ | <../ndarray/sparse.rst#mxnet.ndarray.sparse.round>`_.                                      |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.rint                                                | Convenience fluent method for `rint() <../ndarray/sparse.rst#mxnet.ndarray.sparse.rint>`_. |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.fix                                                 | Convenience fluent method for `fix() <../ndarray/sparse.rst#mxnet.ndarray.sparse.fix>`_.   |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.floor                                               | Convenience fluent method for `floor()                                                     |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray.floor>`_ | <../ndarray/sparse.rst#mxnet.ndarray.sparse.floor>`_.                                      |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.ceil                                                | Convenience fluent method for `ceil() <../ndarray/sparse.rst#mxnet.ndarray.sparse.ceil>`_. |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.trunc                                               | Convenience fluent method for `trunc()                                                     |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray.trunc>`_ | <../ndarray/sparse.rst#mxnet.ndarray.sparse.trunc>`_.                                      |
+-----------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Indexing
--------

+-----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.__getitem__                                               | x.__getitem__(i) <=> x[i]                                                                  |
+-----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.__setitem__                                               | x.__setitem__(i, y) <=> x[i]=y                                                             |
+-----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Lazy evaluation
---------------

+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `RowSparseNDArray.wait_to_read                                               | Waits until all previous write operations on the current array are finished.               |
+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array creation routines
=======================

+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `array                                                          | Creates a sparse array from any object exposing the array interface.                       |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `empty                                                          | Returns a new array of given shape and type, without initializing entries.                 |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `zeros                                                          | Return a new array of given shape and type, filled with zeros.                             |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `zeros_like                                                     | Return an array of zeros with the same shape and type as the input array.                  |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `csr_matrix                                                     | Creates a 2D array with compressed sparse row (CSR) format.                                |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `row_sparse_array                                               | Creates a multidimensional row sparse array with a set of tensor slices at given indices.  |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `mxnet.ndarray.load                                             | Loads an array from file.                                                                  |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `mxnet.ndarray.save                                             | Saves a list of arrays or a dict of str->array to file.                                    |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Array manipulation routines
===========================


Changing array storage type
---------------------------

+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `cast_storage                                               | Casts tensor storage type to the new type.                                                 |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Indexing routines
-----------------

+-------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `slice                                                | Slices a contiguous region of the array.                                                   |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `retain                                               | pick rows specified by user input index array from a row sparse matrix                     |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------+


Mathematical functions
======================


Arithmetic operations
---------------------

+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `elemwise_add                                               | Adds arguments element-wise.                                                               |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `elemwise_sub                                               | Subtracts arguments element-wise.                                                          |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `elemwise_mul                                               | Multiplies arguments element-wise.                                                         |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `negative                                                   | Numerical negative of the argument, element-wise.                                          |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `dot                                                        | Dot product of two arrays.                                                                 |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `add_n                                                      | Adds all input arguments element-wise.                                                     |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------+


Trigonometric functions
-----------------------

+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `sin                                                   | Computes the element-wise sine of the input array.                                         |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `tan                                                   | Computes the element-wise tangent of the input array.                                      |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `arcsin                                                | Returns element-wise inverse sine of the input array.                                      |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `arctan                                                | Returns element-wise inverse tangent of the input array.                                   |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `degrees                                               | Converts each element of the input array from radians to degrees.                          |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `radians                                               | Converts each element of the input array from degrees to radians.                          |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+


Hyperbolic functions
--------------------

+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `sinh                                                  | Returns the hyperbolic sine of the input array, computed element-wise.                     |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `tanh                                                  | Returns the hyperbolic tangent of the input array, computed element-wise.                  |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `arcsinh                                               | Returns the element-wise inverse hyperbolic sine of the input array, computed              |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.arcsinh>`_ | element-wise.                                                                              |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `arctanh                                               | Returns the element-wise inverse hyperbolic tangent of the input array, computed           |
| <../ndarray/sparse.rst#mxnet.ndarray.sparse.arctanh>`_ | element-wise.                                                                              |
+--------------------------------------------------------+--------------------------------------------------------------------------------------------+


Rounding
--------

+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `round                                               | Returns element-wise rounded value to the nearest integer of the input.                    |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `rint                                                | Returns element-wise rounded value to the nearest integer of the input.                    |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `fix                                                 | Returns element-wise rounded value to the nearest integer towards zero of the input.       |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `floor                                               | Returns element-wise floor of the input.                                                   |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `ceil                                                | Returns element-wise ceiling of the input.                                                 |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `trunc                                               | Return the element-wise truncated value of the input.                                      |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+


Exponents and logarithms
------------------------

+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `expm1                                               | Returns ``exp(x) - 1`` computed element-wise on the input.                                 |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `log1p                                               | Returns element-wise ``log(1 + x)`` value of the input.                                    |
+------------------------------------------------------+--------------------------------------------------------------------------------------------+


Powers
------

+-------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `sqrt                                                 | Returns element-wise square-root value of the input.                                       |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `square                                               | Returns element-wise squared value of the input.                                           |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------+


Miscellaneous
-------------

+-----------------------------------------------------+--------------------------------------------------------------------------------------------+
| `abs                                                | Returns element-wise absolute value of the input.                                          |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------+
| `sign                                               | Returns element-wise sign of the input.                                                    |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------+


More
----

+--------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `make_loss                                                   | Stops gradient computation.                                                                |
+--------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `stop_gradient                                               | Stops gradient computation.                                                                |
+--------------------------------------------------------------+--------------------------------------------------------------------------------------------+


API Reference
=============

**class mxnet.ndarray.sparse.CSRNDArray(handle, writable=True)**

   A sparse representation of 2D NDArray in the Compressed Sparse Row
   format.

   A CSRNDArray represents an NDArray as three separate arrays:
   *data*, *indptr* and *indices*. It uses the standard CSR
   representation where the column indices for row i are stored in
   ``indices[indptr[i]:indptr[i+1]]`` and their corresponding values
   are stored in ``data[indptr[i]:indptr[i+1]]``.

   The column indices for a given row are expected to be sorted in
   ascending order. Duplicate column entries for the same row are not
   allowed.

   -[ Example ]-

   >>> a = mx.nd.array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]])
   >>> a = a.tostype('csr')
   >>> a.indices.asnumpy()
   array([1, 0, 2])
   >>> a.indptr.asnumpy()
   array([0, 1, 2, 2, 3])
   >>> a.data.asnumpy()
   array([ 1.,  2.,  3.], dtype=float32)

   **__getitem__(key)**

      x.__getitem__(i) <=> x[i]

      Returns a sliced view of this array.

      :Parameters:
         **key** (`slice
         <../symbol/symbol.rst#mxnet.symbol.Symbol.slice>`_) --
         Indexing key.

      -[ Examples ]-

      >>> indptr = np.array([0, 2, 3, 6])
      >>> indices = np.array([0, 2, 2, 0, 1, 2])
      >>> data = np.array([1, 2, 3, 4, 5, 6])
      >>> a = mx.nd.sparse.csr_matrix(data, indptr, indices, (3, 3))
      >>> a.asnumpy()
      array([[1, 0, 2],
             [0, 0, 3],
             [4, 5, 6]])
      >>> a[1:2].asnumpy()
      array([[0, 0, 3]], dtype=float32)

   **__setitem__(key, value)**

      x.__setitem__(i, y) <=> x[i]=y

      Set self[key] to value. Only slice key [:] is supported.

      :Parameters:
         * **key** (`slice
           <../symbol/symbol.rst#mxnet.symbol.Symbol.slice>`_) -- The
           indexing key.

         * **value** (`NDArray
           <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_* or
           *`CSRNDArray
           <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_*
           or **numpy.ndarray*) -- The value to set.

      -[ Examples ]-

      >>> src = mx.nd.sparse.zeros('csr', (3,3))
      >>> src.asnumpy()
      array([[ 0.,  0.,  0.],
             [ 0.,  0.,  0.],
             [ 0.,  0.,  0.]], dtype=float32)
      >>> # assign CSRNDArray with same storage type
      >>> x = mx.nd.ones('row_sparse', (3,3)).tostype('csr')
      >>> x[:] = src
      >>> x.asnumpy()
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)
      >>> # assign NDArray to CSRNDArray
      >>> x[:] = mx.nd.ones((3,3)) * 2
      >>> x.asnumpy()
      array([[ 2.,  2.,  2.],
             [ 2.,  2.,  2.],
             [ 2.,  2.,  2.]], dtype=float32)

   ``indices``

      A deep copy NDArray of the indices array of the CSRNDArray. This
      generates a deep copy of the column indices of the current *csr*
      matrix.

      :Returns:
         This CSRNDArray's indices array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_

   ``indptr``

      A deep copy NDArray of the indptr array of the CSRNDArray. This
      generates a deep copy of the *indptr* of the current *csr*
      matrix.

      :Returns:
         This CSRNDArray's indptr array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_

   ``data``

      A deep copy NDArray of the data array of the CSRNDArray. This
      generates a deep copy of the *data* of the current *csr* matrix.

      :Returns:
         This CSRNDArray's data array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_

   **tostype(stype)**

      Return a copy of the array with chosen storage type.

      :Returns:
         A copy of the array with the chosen storage stype

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
         `CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_

   **copyto(other)**

      Copies the value of this array to another array.

      If ``other`` is a ``NDArray`` or ``CSRNDArray`` object, then
      ``other.shape`` and ``self.shape`` should be the same. This
      function copies the value from ``self`` to ``other``.

      If ``other`` is a context, a new ``CSRNDArray`` will be first
      created on the target context, and the value of ``self`` is
      copied.

      :Parameters:
         **other** (`NDArray
         <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_* or
         *`CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_* or
         **Context*) -- The destination array or context.

      :Returns:
         The copied array. If ``other`` is an ``NDArray`` or
         ``CSRNDArray``, then the return value and ``other`` will
         point to the same ``NDArray`` or ``CSRNDArray``.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
         `CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_

   **as_in_context(context)**

      Returns an array on the target device with the same value as
      this array.

      If the target context is the same as ``self.context``, then
      ``self`` is returned.  Otherwise, a copy is made.

      :Parameters:
         **context** (*Context*) -- The target context.

      :Returns:
         The target array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_,
         `CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_ or
         `RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

      -[ Examples ]-

      >>> x = mx.nd.ones((2,3))
      >>> y = x.as_in_context(mx.cpu())
      >>> y is x
      True
      >>> z = x.as_in_context(mx.gpu(0))
      >>> z is x
      False

   **asnumpy()**

      Return a dense ``numpy.ndarray`` object with value copied from
      this array

   **asscalar()**

      Returns a scalar whose value is copied from this array.

      This function is equivalent to ``self.asnumpy()[0]``. This
      NDArray must have shape (1,).

      -[ Examples ]-

      >>> x = mx.nd.ones((1,), dtype='int32')
      >>> x.asscalar()
      1
      >>> type(x.asscalar())
      <type 'numpy.int32'>

   **astype(dtype)**

      Returns a copy of the array after casting to a specified type.
      :param dtype: The type of the returned array. :type dtype:
      numpy.dtype or str

      -[ Examples ]-

      >>> x = mx.nd.sparse.zeros('row_sparse', (2,3), dtype='float32')
      >>> y = x.astype('int32')
      >>> y.dtype
      <type 'numpy.int32'>

   ``context``

      Device context of the array.

      -[ Examples ]-

      >>> x = mx.nd.array([1, 2, 3, 4])
      >>> x.context
      cpu(0)
      >>> type(x.context)
      <class 'mxnet.context.Context'>
      >>> y = mx.nd.zeros((2,3), mx.gpu(0))
      >>> y.context
      gpu(0)

   **copy()**

      Makes a copy of this ``NDArray``, keeping the same context.

      :Returns:
         The copied array

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_,
         `CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_ or
         `RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

      -[ Examples ]-

      >>> x = mx.nd.ones((2,3))
      >>> y = x.copy()
      >>> y.asnumpy()
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)

   ``dtype``

      Data-type of the array's elements.

      :Returns:
         This NDArray's data type.

      :Return type:
         numpy.dtype

      -[ Examples ]-

      >>> x = mx.nd.zeros((2,3))
      >>> x.dtype
      <type 'numpy.float32'>
      >>> y = mx.nd.zeros((2,3), dtype='int32')
      >>> y.dtype
      <type 'numpy.int32'>

   ``shape``

      Tuple of array dimensions.

      -[ Examples ]-

      >>> x = mx.nd.array([1, 2, 3, 4])
      >>> x.shape
      (4L,)
      >>> y = mx.nd.zeros((2, 3, 4))
      >>> y.shape
      (2L, 3L, 4L)

   **slice(*args, **kwargs)**

      Convenience fluent method for `slice()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.slice>`_.

      The arguments are the same as for `slice()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.slice>`_, with this
      array as data.

   ``stype``

      Storage-type of the array.

   **wait_to_read()**

      Waits until all previous write operations on the current array
      are finished.

      This method guarantees that all previous write operations that
      pushed into the backend engine for execution are actually
      finished.

      -[ Examples ]-

      >>> import time
      >>> tic = time.time()
      >>> a = mx.nd.ones((1000,1000))
      >>> b = mx.nd.dot(a, a)
      >>> print(time.time() - tic) # doctest: +SKIP
      0.003854036331176758
      >>> b.wait_to_read()
      >>> print(time.time() - tic) # doctest: +SKIP
      0.0893700122833252

   **zeros_like(*args, **kwargs)**

      Convenience fluent method for `zeros_like()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.zeros_like>`_.

      The arguments are the same as for `zeros_like()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.zeros_like>`_, with
      this array as data.

**class mxnet.ndarray.sparse.RowSparseNDArray(handle, writable=True)**

   A sparse representation of a set of NDArray row slices at given
   indices.

   A RowSparseNDArray represents a multidimensional NDArray using two
   separate arrays: *data* and *indices*.

   * data: an NDArray of any dtype with shape [D0, D1, ..., Dn].

   * indices: a 1-D int64 NDArray with shape [D0] with values sorted
     in ascending order.

   The *indices* stores the indices of the row slices with non-zeros,
   while the values are stored in *data*. The corresponding NDArray
   ``dense`` represented by RowSparseNDArray ``rsp`` has

   ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

   >>> dense.asnumpy()
   array([[ 1.,  2., 3.],
          [ 0.,  0., 0.],
          [ 4.,  0., 5.],
          [ 0.,  0., 0.],
          [ 0.,  0., 0.]], dtype=float32)
   >>> rsp = dense.tostype('row_sparse')
   >>> rsp.indices.asnumpy()
   array([0, 2], dtype=int64)
   >>> rsp.data.asnumpy()
   array([[ 1.,  2., 3.],
          [ 4.,  0., 5.]], dtype=float32)

   A RowSparseNDArray is typically used to represent non-zero row
   slices of a large NDArray of shape [LARGE0, D1, .. , Dn] where
   LARGE0 >> D0 and most row slices are zeros.

   RowSparseNDArray is used principally in the definition of gradients
   for operations that have sparse gradients (e.g. sparse dot and
   sparse embedding).

   **__getitem__(key)**

      x.__getitem__(i) <=> x[i]

      Returns a sliced view of this array.

      :Parameters:
         **key** (`slice
         <../symbol/symbol.rst#mxnet.symbol.Symbol.slice>`_) --
         Indexing key.

      -[ Examples ]-

      >>> x = mx.nd.sparse.zeros('row_sparse', (2, 3))
      >>> x[:].asnumpy()
      array([[ 0.,  0.,  0.],
             [ 0.,  0.,  0.]], dtype=float32)

   **__setitem__(key, value)**

      x.__setitem__(i, y) <=> x[i]=y

      Set self[key] to value. Only slice key [:] is supported.

      :Parameters:
         * **key** (`slice
           <../symbol/symbol.rst#mxnet.symbol.Symbol.slice>`_) -- The
           indexing key.

         * **value** (`NDArray
           <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_* or
           **numpy.ndarray*) -- The value to set.

      -[ Examples ]-

      >>> src = mx.nd.row_sparse([[1, 0, 2], [4, 5, 6]], [0, 2], (3,3))
      >>> src.asnumpy()
      array([[ 1.,  0.,  2.],
             [ 0.,  0.,  0.],
             [ 4.,  5.,  6.]], dtype=float32)
      >>> # assign RowSparseNDArray with same storage type
      >>> x = mx.nd.sparse.zeros('row_sparse', (3,3))
      >>> x[:] = src
      >>> x.asnumpy()
      array([[ 1.,  0.,  2.],
             [ 0.,  0.,  0.],
             [ 4.,  5.,  6.]], dtype=float32)
      >>> # assign NDArray to RowSparseNDArray
      >>> x[:] = mx.nd.ones((3,3))
      >>> x.asnumpy()
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)

   ``indices``

      A deep copy NDArray of the indices array of the
      RowSparseNDArray. This generates a deep copy of the row indices
      of the current *row_sparse* matrix.

      :Returns:
         This RowSparseNDArray's indices array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_

   ``data``

      A deep copy NDArray of the data array of the RowSparseNDArray.
      This generates a deep copy of the *data* of the current
      *row_sparse* matrix.

      :Returns:
         This RowSparseNDArray's data array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_

   **tostype(stype)**

      Return a copy of the array with chosen storage type.

      :Returns:
         A copy of the array with the chosen storage stype

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
         `RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

   **copyto(other)**

      Copies the value of this array to another array.

      If ``other`` is a ``NDArray`` or ``RowSparseNDArray`` object,
      then ``other.shape`` and ``self.shape`` should be the same. This
      function copies the value from ``self`` to ``other``.

      If ``other`` is a context, a new ``RowSparseNDArray`` will be
      first created on the target context, and the value of ``self``
      is copied.

      :Parameters:
         **other** (`NDArray
         <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_* or
         *`RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_*
         or **Context*) -- The destination array or context.

      :Returns:
         The copied array. If ``other`` is an ``NDArray`` or
         ``RowSparseNDArray``, then the return value and ``other``
         will point to the same ``NDArray`` or ``RowSparseNDArray``.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
         `RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

   **as_in_context(context)**

      Returns an array on the target device with the same value as
      this array.

      If the target context is the same as ``self.context``, then
      ``self`` is returned.  Otherwise, a copy is made.

      :Parameters:
         **context** (*Context*) -- The target context.

      :Returns:
         The target array.

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_,
         `CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_ or
         `RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

      -[ Examples ]-

      >>> x = mx.nd.ones((2,3))
      >>> y = x.as_in_context(mx.cpu())
      >>> y is x
      True
      >>> z = x.as_in_context(mx.gpu(0))
      >>> z is x
      False

   **asnumpy()**

      Return a dense ``numpy.ndarray`` object with value copied from
      this array

   **asscalar()**

      Returns a scalar whose value is copied from this array.

      This function is equivalent to ``self.asnumpy()[0]``. This
      NDArray must have shape (1,).

      -[ Examples ]-

      >>> x = mx.nd.ones((1,), dtype='int32')
      >>> x.asscalar()
      1
      >>> type(x.asscalar())
      <type 'numpy.int32'>

   **astype(dtype)**

      Returns a copy of the array after casting to a specified type.
      :param dtype: The type of the returned array. :type dtype:
      numpy.dtype or str

      -[ Examples ]-

      >>> x = mx.nd.sparse.zeros('row_sparse', (2,3), dtype='float32')
      >>> y = x.astype('int32')
      >>> y.dtype
      <type 'numpy.int32'>

   **ceil(*args, **kwargs)**

      Convenience fluent method for `ceil()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.ceil>`_.

      The arguments are the same as for `ceil()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.ceil>`_, with this
      array as data.

   ``context``

      Device context of the array.

      -[ Examples ]-

      >>> x = mx.nd.array([1, 2, 3, 4])
      >>> x.context
      cpu(0)
      >>> type(x.context)
      <class 'mxnet.context.Context'>
      >>> y = mx.nd.zeros((2,3), mx.gpu(0))
      >>> y.context
      gpu(0)

   **copy()**

      Makes a copy of this ``NDArray``, keeping the same context.

      :Returns:
         The copied array

      :Return type:
         `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_,
         `CSRNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_ or
         `RowSparseNDArray
         <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

      -[ Examples ]-

      >>> x = mx.nd.ones((2,3))
      >>> y = x.copy()
      >>> y.asnumpy()
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)

   ``dtype``

      Data-type of the array's elements.

      :Returns:
         This NDArray's data type.

      :Return type:
         numpy.dtype

      -[ Examples ]-

      >>> x = mx.nd.zeros((2,3))
      >>> x.dtype
      <type 'numpy.float32'>
      >>> y = mx.nd.zeros((2,3), dtype='int32')
      >>> y.dtype
      <type 'numpy.int32'>

   **fix(*args, **kwargs)**

      Convenience fluent method for `fix()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.fix>`_.

      The arguments are the same as for `fix()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.fix>`_, with this
      array as data.

   **floor(*args, **kwargs)**

      Convenience fluent method for `floor()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.floor>`_.

      The arguments are the same as for `floor()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.floor>`_, with this
      array as data.

   **rint(*args, **kwargs)**

      Convenience fluent method for `rint()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.rint>`_.

      The arguments are the same as for `rint()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.rint>`_, with this
      array as data.

   **round(*args, **kwargs)**

      Convenience fluent method for `round()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.round>`_.

      The arguments are the same as for `round()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.round>`_, with this
      array as data.

   ``shape``

      Tuple of array dimensions.

      -[ Examples ]-

      >>> x = mx.nd.array([1, 2, 3, 4])
      >>> x.shape
      (4L,)
      >>> y = mx.nd.zeros((2, 3, 4))
      >>> y.shape
      (2L, 3L, 4L)

   ``stype``

      Storage-type of the array.

   **trunc(*args, **kwargs)**

      Convenience fluent method for `trunc()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.trunc>`_.

      The arguments are the same as for `trunc()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.trunc>`_, with this
      array as data.

   **wait_to_read()**

      Waits until all previous write operations on the current array
      are finished.

      This method guarantees that all previous write operations that
      pushed into the backend engine for execution are actually
      finished.

      -[ Examples ]-

      >>> import time
      >>> tic = time.time()
      >>> a = mx.nd.ones((1000,1000))
      >>> b = mx.nd.dot(a, a)
      >>> print(time.time() - tic) # doctest: +SKIP
      0.003854036331176758
      >>> b.wait_to_read()
      >>> print(time.time() - tic) # doctest: +SKIP
      0.0893700122833252

   **zeros_like(*args, **kwargs)**

      Convenience fluent method for `zeros_like()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.zeros_like>`_.

      The arguments are the same as for `zeros_like()
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.zeros_like>`_, with
      this array as data.

Sparse NDArray API of MXNet.

**mxnet.ndarray.sparse.csr_matrix(data, indptr, indices, shape,
ctx=None, dtype=None, indptr_type=None, indices_type=None)**

   Creates a 2D array with compressed sparse row (CSR) format.

   :Parameters:
      * **data** (*array_like*) -- An object exposing the array
        interface, with shape [nnz], where D0 is the number of
        non-zero entries.

      * **indptr** (*array_like*) -- An object exposing the array
        interface, with shape [D0 + 1]. The first element in indptr
        should always be zero.

      * **indices** (*array_like*) -- An object exposing the array
        interface, with shape [nnz].

      * **ctx** (*Context**, **optional*) -- Device context (default
        is the current default context).

      * **dtype** (*str** or **numpy.dtype**, **optional*) -- The data
        type of the output array. The default dtype is
        ``values.dtype`` if *values* is an *NDArray*, *float32*
        otherwise.

      * **indptr_type** (*str** or **numpy.dtype**, **optional*) --
        The data type of the indices array. The default dtype is
        ``indptr.dtype`` if *indptr* is an *NDArray*, *int64*
        otherwise.

      * **indices_type** (*str** or **numpy.dtype**, **optional*) --
        The data type of the indices array. The default dtype is
        ``indices.dtype`` if *indicies* is an *NDArray*, *int64*
        otherwise.

   :Returns:
      A *CSRNDArray* with the *csr* storage representation.

   :Return type:
      `CSRNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_

   -[ Example ]-

   >>> import mxnet as mx
   >>> a = mx.nd.sparse.csr_matrix([1, 2, 3], [0, 1, 2, 2, 3], [1, 0, 2], (4, 3))
   >>> a.asnumpy()
   array([[ 0.,  1.,  0.],
          [ 2.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  3.]], dtype=float32)

**mxnet.ndarray.sparse.row_sparse_array(data, indices, shape,
ctx=None, dtype=None, indices_type=None)**

   Creates a multidimensional row sparse array with a set of tensor
   slices at given indices.

   :Parameters:
      * **data** (*array_like*) -- An object exposing the array
        interface, with shape [D0, D1, .. DK], where D0 is the number
        of rows with non-zeros entries.

      * **indices** (*array_like*) -- An object exposing the array
        interface, with shape [D0].

      * **ctx** (*Context**, **optional*) -- Device context (default
        is the current default context).

      * **dtype** (*str** or **numpy.dtype**, **optional*) -- The data
        type of the output array. The default dtype is ``data.dtype``
        if *data* is an *NDArray*, *float32* otherwise.

      * **indices_type** (*str** or **numpy.dtype**, **optional*) --
        The data type of the indices array. The default dtype is
        ``indices.dtype`` if *indicies* is an *NDArray*, *int64*
        otherwise.

   :Returns:
      An *RowSparseNDArray* with the *row_sparse* storage
      representation.

   :Return type:
      `RowSparseNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

   -[ Example ]-

   >>> a = mx.nd.sparse.row_sparse_array([[1, 2], [3, 4]], [1, 4], (6, 2))
   >>> a.asnumpy()
   array([[ 0.,  0.],
          [ 1.,  2.],
          [ 0.,  0.],
          [ 0.,  0.],
          [ 3.,  4.],
          [ 0.,  0.]], dtype=float32)

**mxnet.ndarray.sparse.ElementWiseSum(*args, **kwargs)**

   Adds all input arguments element-wise.

      add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

   ``add_n`` is potentially more efficient than calling ``add`` by *n*
   times.

   The storage type of ``add_n`` output depends on storage types of
   inputs

   * add_n(row_sparse, row_sparse, ..) = row_sparse

   * otherwise, ``add_n`` generates output with default storage

   Defined in src/operator/tensor/elemwise_sum.cc:L122

   :Parameters:
      * **args** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*[****]***) --
        Positional input arguments

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.abs(data=None, out=None, name=None, **kwargs)**

   Returns element-wise absolute value of the input.

   Example:

   ::

      abs([-2, 0, 3]) = [2, 0, 3]

   The storage type of ``abs`` output depends upon the input storage
   type:

   ..

      * abs(default) = default

      * abs(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L293

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.add_n(*args, **kwargs)**

   Adds all input arguments element-wise.

      add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

   ``add_n`` is potentially more efficient than calling ``add`` by *n*
   times.

   The storage type of ``add_n`` output depends on storage types of
   inputs

   * add_n(row_sparse, row_sparse, ..) = row_sparse

   * otherwise, ``add_n`` generates output with default storage

   Defined in src/operator/tensor/elemwise_sum.cc:L122

   :Parameters:
      * **args** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*[****]***) --
        Positional input arguments

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.arccos(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise inverse cosine of the input array.

   The input should be in range *[-1, 1]*. The output is in the closed
   interval [0, \pi]

      arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]

   The storage type of ``arccos`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L712

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.arccosh(data=None, out=None, name=None,
**kwargs)**

   Returns the element-wise inverse hyperbolic cosine of the input
   array, computed element-wise.

   The storage type of ``arccosh`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L853

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.arcsin(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise inverse sine of the input array.

   The input should be in the range *[-1, 1]*. The output is in the
   closed interval of [-\pi/2, \pi/2].

      arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4,
      \pi/2]

   The storage type of ``arcsin`` output depends upon the input
   storage type:

   ..

      * arcsin(default) = default

      * arcsin(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L693

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.arcsinh(data=None, out=None, name=None,
**kwargs)**

   Returns the element-wise inverse hyperbolic sine of the input
   array, computed element-wise.

   The storage type of ``arcsinh`` output depends upon the input
   storage type:

   ..

      * arcsinh(default) = default

      * arcsinh(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L839

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.arctan(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise inverse tangent of the input array.

   The output is in the closed interval [-\pi/2, \pi/2]

      arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

   The storage type of ``arctan`` output depends upon the input
   storage type:

   ..

      * arctan(default) = default

      * arctan(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L733

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.arctanh(data=None, out=None, name=None,
**kwargs)**

   Returns the element-wise inverse hyperbolic tangent of the input
   array, computed element-wise.

   The storage type of ``arctanh`` output depends upon the input
   storage type:

   ..

      * arctanh(default) = default

      * arctanh(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L870

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.cast_storage(data=None, stype=_Null, out=None,
name=None, **kwargs)**

   Casts tensor storage type to the new type.

   When an NDArray with default storage type is cast to csr or
   row_sparse storage, the result is compact, which means:

   * for csr, zero values will not be retained

   * for row_sparse, row slices of all zeros will not be retained

   The storage type of ``cast_storage`` output depends on stype
   parameter:

   * cast_storage(csr, 'default') = default

   * cast_storage(row_sparse, 'default') = default

   * cast_storage(default, 'csr') = csr

   * cast_storage(default, 'row_sparse') = row_sparse

   Example:

   ::

      dense = [[ 0.,  1.,  0.],
               [ 2.,  0.,  3.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]]

      # cast to row_sparse storage type
      rsp = cast_storage(default, 'row_sparse')
      rsp.indices = [0, 1]
      rsp.values = [[ 0.,  1.,  0.],
                    [ 2.,  0.,  3.]]

      # cast to csr storage type
      csr = cast_storage(default, 'csr')
      csr.indices = [1, 0, 2]
      csr.values = [ 1.,  2.,  3.]
      csr.indptr = [0, 1, 3, 3, 3]

   Defined in src/operator/tensor/cast_storage.cc:L69

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The
        input.

      * **stype** (*{'csr'**, **'default'**, **'row_sparse'}**,
        **required*) -- Output storage type.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.ceil(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise ceiling of the input.

   The ceil of the scalar x is the smallest integer i, such that i >=
   x.

   Example:

   ::

      ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]

   The storage type of ``ceil`` output depends upon the input storage
   type:

   ..

      * ceil(default) = default

      * ceil(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L370

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.cos(data=None, out=None, name=None, **kwargs)**

   Computes the element-wise cosine of the input array.

   The input should be in radians (2\pi rad equals 360 degrees).

      cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]

   The storage type of ``cos`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L652

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.cosh(data=None, out=None, name=None,
**kwargs)**

   Returns the hyperbolic cosine  of the input array, computed
   element-wise.

      cosh(x) = 0.5\times(exp(x) + exp(-x))

   The storage type of ``cosh`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L805

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.degrees(data=None, out=None, name=None,
**kwargs)**

   Converts each element of the input array from radians to degrees.

      degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

   The storage type of ``degrees`` output depends upon the input
   storage type:

   ..

      * degrees(default) = default

      * degrees(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L752

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.dot(lhs=None, rhs=None, transpose_a=_Null,
transpose_b=_Null, out=None, name=None, **kwargs)**

   Dot product of two arrays.

   ``dot``'s behavior depends on the input array dimensions:

   * 1-D arrays: inner product of vectors

   * 2-D arrays: matrix multiplication

   * N-D arrays: a sum product over the last axis of the first input
     and the first axis of the second input

     For example, given 3-D ``x`` with shape *(n,m,k)* and ``y`` with
     shape *(k,r,s)*, the result array will have shape *(n,m,r,s)*. It
     is computed by:

     ::
        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])

     Example:

     ::
        x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
        y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
        dot(x,y)[0,0,1,1] = 0
        sum(x[0,0,:]*y[:,1,1]) = 0

   The storage type of ``dot`` output depends on storage types of
   inputs and transpose options:

   * dot(csr, default) = default

   * dot(csr.T, default) = row_sparse

   * dot(csr, row_sparse) = default

   * otherwise, ``dot`` generates output with default storage

   Defined in src/operator/tensor/dot.cc:L61

   :Parameters:
      * **lhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The first
        input

      * **rhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The
        second input

      * **transpose_a** (*boolean**, **optional**, **default=False*)
        -- If true then transpose the first input before dot.

      * **transpose_b** (*boolean**, **optional**, **default=False*)
        -- If true then transpose the second input before dot.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.elemwise_add(lhs=None, rhs=None, out=None,
name=None, **kwargs)**

   Adds arguments element-wise.

   The storage type of ``elemwise_add`` output depends on storage
   types of inputs

   ..

      * elemwise_add(row_sparse, row_sparse) = row_sparse

      * otherwise, ``elemwise_add`` generates output with default
        storage

   :Parameters:
      * **lhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- first
        input

      * **rhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- second
        input

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.elemwise_div(lhs=None, rhs=None, out=None,
name=None, **kwargs)**

   Divides arguments element-wise.

   The storage type of ``elemwise_dev`` output is always dense

   :Parameters:
      * **lhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- first
        input

      * **rhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- second
        input

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.elemwise_mul(lhs=None, rhs=None, out=None,
name=None, **kwargs)**

   Multiplies arguments element-wise.

   The storage type of ``elemwise_mul`` output depends on storage
   types of inputs

   ..

      * elemwise_mul(default, default) = default

      * elemwise_mul(row_sparse, row_sparse) = row_sparse

      * elemwise_mul(default, row_sparse) = row_sparse

      * elemwise_mul(row_sparse, default) = row_sparse

      * otherwise, ``elemwise_mul`` generates output with default
        storage

   :Parameters:
      * **lhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- first
        input

      * **rhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- second
        input

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.elemwise_sub(lhs=None, rhs=None, out=None,
name=None, **kwargs)**

   Subtracts arguments element-wise.

   The storage type of ``elemwise_sub`` output depends on storage
   types of inputs

   ..

      * elemwise_sub(row_sparse, row_sparse) = row_sparse

      * otherwise, ``elemwise_add`` generates output with default
        storage

   :Parameters:
      * **lhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- first
        input

      * **rhs** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- second
        input

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.exp(data=None, out=None, name=None, **kwargs)**

   Returns element-wise exponential value of the input.

      exp(x) = e^x \approx 2.718^x

   Example:

   ::

      exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]

   The storage type of ``exp`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L543

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.expm1(data=None, out=None, name=None,
**kwargs)**

   Returns ``exp(x) - 1`` computed element-wise on the input.

   This function provides greater precision than ``exp(x) - 1`` for
   small values of ``x``.

   The storage type of ``expm1`` output depends upon the input storage
   type:

   ..

      * expm1(default) = default

      * expm1(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L635

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.fix(data=None, out=None, name=None, **kwargs)**

   Returns element-wise rounded value to the nearest integer towards
   zero of the input.

   Example:

   ::

      fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]

   The storage type of ``fix`` output depends upon the input storage
   type:

   ..

      * fix(default) = default

      * fix(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L424

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.floor(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise floor of the input.

   The floor of the scalar x is the largest integer i, such that i <=
   x.

   Example:

   ::

      floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]

   The storage type of ``floor`` output depends upon the input storage
   type:

   ..

      * floor(default) = default

      * floor(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L388

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.gamma(data=None, out=None, name=None,
**kwargs)**

   Returns the gamma function (extension of the factorial function to
   the reals), computed element-wise on the input array.

   The storage type of ``gamma`` output is always dense

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.gammaln(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise log of the absolute value of the gamma
   function of the input.

   The storage type of ``gammaln`` output is always dense

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.log(data=None, out=None, name=None, **kwargs)**

   Returns element-wise Natural logarithmic value of the input.

   The natural logarithm is logarithm in base *e*, so that
   ``log(exp(x)) = x``

   The storage type of ``log`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L555

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.log10(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise Base-10 logarithmic value of the input.

   ``10**log10(x) = x``

   The storage type of ``log10`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L567

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.log1p(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise ``log(1 + x)`` value of the input.

   This function is more accurate than ``log(1 + x)``  for small ``x``
   so that 1+x\approx 1

   The storage type of ``log1p`` output depends upon the input storage
   type:

   ..

      * log1p(default) = default

      * log1p(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L617

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.log2(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise Base-2 logarithmic value of the input.

   ``2**log2(x) = x``

   The storage type of ``log2`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L579

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.make_loss(data=None, out=None, name=None,
**kwargs)**

   Stops gradient computation. .. note:: ``make_loss`` is deprecated,
   use ``MakeLoss``.

   The storage type of ``make_loss`` output depends upon the input
   storage type:

   ..

      * make_loss(default) = default

      * make_loss(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L148

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.negative(data=None, out=None, name=None,
**kwargs)**

   Numerical negative of the argument, element-wise.

   The storage type of ``negative`` output depends upon the input
   storage type:

   ..

      * negative(default) = default

      * negative(row_sparse) = row_sparse

      * negative(csr) = csr

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.radians(data=None, out=None, name=None,
**kwargs)**

   Converts each element of the input array from degrees to radians.

      radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

   The storage type of ``radians`` output depends upon the input
   storage type:

   ..

      * radians(default) = default

      * radians(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L771

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.relu(data=None, out=None, name=None,
**kwargs)**

   Computes rectified linear.

      max(features, 0)

   The storage type of ``relu`` output depends upon the input storage
   type:

   ..

      * relu(default) = default

      * relu(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L44

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.retain(data=None, indices=None, out=None,
name=None, **kwargs)**

   pick rows specified by user input index array from a row sparse
   matrix and save them in the output sparse matrix.

   Example:

   ::

      data = [[1, 2], [3, 4], [5, 6]]
      indices = [0, 1, 3]
      shape = (4, 2)
      rsp_in = row_sparse(data, indices)
      to_retain = [0, 3]
      rsp_out = retain(rsp_in, to_retain)
      rsp_out.values = [[1, 2], [5, 6]]
      rsp_out.indices = [0, 3]

   The storage type of ``retain`` output depends on storage types of
   inputs

   * retain(row_sparse, default) = row_sparse

   * otherwise, ``retain`` is not supported

   Defined in src/operator/tensor/sparse_retain.cc:L53

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array for sparse_retain operator.

      * **indices** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The index
        array of rows ids that will be retained.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.rint(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise rounded value to the nearest integer of the
   input.

   Note: * For input ``n.5`` ``rint`` returns ``n`` while ``round``
       returns ``n+1``.

     * For input ``-n.5`` both ``rint`` and ``round`` returns
       ``-n-1``.

   Example:

   ::

      rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]

   The storage type of ``rint`` output depends upon the input storage
   type:

   ..

      * rint(default) = default

      * rint(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L352

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.round(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise rounded value to the nearest integer of the
   input.

   Example:

   ::

      round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]

   The storage type of ``round`` output depends upon the input storage
   type:

   ..

      * round(default) = default

      * round(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L331

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.rsqrt(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise inverse square-root value of the input.

      rsqrt(x) = 1/\sqrt{x}

   Example:

   ::

      rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]

   The storage type of ``rsqrt`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L487

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.sigmoid(data=None, out=None, name=None,
**kwargs)**

   Computes sigmoid of x element-wise.

      y = 1 / (1 + exp(-x))

   The storage type of ``sigmoid`` output is always dense

   Defined in src/operator/tensor/elemwise_unary_op.cc:L64

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.sign(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise sign of the input.

   Example:

   ::

      sign([-2, 0, 3]) = [-1, 0, 1]

   The storage type of ``sign`` output depends upon the input storage
   type:

   ..

      * sign(default) = default

      * sign(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L312

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.sin(data=None, out=None, name=None, **kwargs)**

   Computes the element-wise sine of the input array.

   The input should be in radians (2\pi rad equals 360 degrees).

      sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

   The storage type of ``sin`` output depends upon the input storage
   type:

   ..

      * sin(default) = default

      * sin(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L599

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.sinh(data=None, out=None, name=None,
**kwargs)**

   Returns the hyperbolic sine of the input array, computed
   element-wise.

      sinh(x) = 0.5\times(exp(x) - exp(-x))

   The storage type of ``sinh`` output depends upon the input storage
   type:

   ..

      * sinh(default) = default

      * sinh(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L790

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.slice(data=None, begin=_Null, end=_Null,
out=None, name=None, **kwargs)**

   Slices a contiguous region of the array.

   Note: ``crop`` is deprecated. Use ``slice`` instead.

   This function returns a sliced continuous region of the array
   between the indices given by *begin* and *end*.

   For an input array of *n* dimensions, slice operation with
   ``begin=(b_0, b_1...b_n-1)`` indices and ``end=(e_1, e_2, ...
   e_n)`` indices will result in an array with the shape ``(e_1-b_0,
   ..., e_n-b_n-1)``.

   The resulting array's *k*-th dimension contains elements from the
   *k*-th dimension of the input array with the open range ``[b_k,
   e_k)``.

   For an input array of non-default storage type(e.g. *csr* or
   *row_sparse*), it only supports slicing on the first dimension.

   Example:

   ::

      x = [[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.]]

      slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
                                         [ 6.,  7.,  8.]]

   Defined in src/operator/tensor/matrix_op.cc:L278

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- Source
        input

      * **begin** (*Shape**(**tuple**)****, **required*) -- starting
        indices for the slice operation, supports negative indices.

      * **end** (*Shape**(**tuple**)****, **required*) -- ending
        indices for the slice operation, supports negative indices.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.sqrt(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise square-root value of the input.

      \textrm{sqrt}(x) = \sqrt{x}

   Example:

   ::

      sqrt([4, 9, 16]) = [2, 3, 4]

   The storage type of ``sqrt`` output depends upon the input storage
   type:

   ..

      * sqrt(default) = default

      * sqrt(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L467

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.square(data=None, out=None, name=None,
**kwargs)**

   Returns element-wise squared value of the input.

      square(x) = x^2

   Example:

   ::

      square([2, 3, 4]) = [4, 9, 16]

   The storage type of ``square`` output depends upon the input
   storage type:

   ..

      * square(default) = default

      * square(row_sparse) = row_sparse

      * square(csr) = csr

   Defined in src/operator/tensor/elemwise_unary_op.cc:L444

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.stop_gradient(data=None, out=None, name=None,
**kwargs)**

   Stops gradient computation.

   Stops the accumulated gradient of the inputs from flowing through
   this operator in the backward direction. In other words, this
   operator prevents the contribution of its inputs to be taken into
   account for computing gradients.

   Example:

   ::

      v1 = [1, 2]
      v2 = [0, 1]
      a = Variable('a')
      b = Variable('b')
      b_stop_grad = stop_gradient(3 * b)
      loss = MakeLoss(b_stop_grad + a)

      executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
      executor.forward(is_train=True, a=v1, b=v2)
      executor.outputs
      [ 1.  5.]

      executor.backward()
      executor.grad_arrays
      [ 0.  0.]
      [ 1.  1.]

   Defined in src/operator/tensor/elemwise_unary_op.cc:L128

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.tan(data=None, out=None, name=None, **kwargs)**

   Computes the element-wise tangent of the input array.

   The input should be in radians (2\pi rad equals 360 degrees).

      tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

   The storage type of ``tan`` output depends upon the input storage
   type:

   ..

      * tan(default) = default

      * tan(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L672

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.tanh(data=None, out=None, name=None,
**kwargs)**

   Returns the hyperbolic tangent of the input array, computed
   element-wise.

      tanh(x) = sinh(x) / cosh(x)

   The storage type of ``tanh`` output depends upon the input storage
   type:

   ..

      * tanh(default) = default

      * tanh(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L823

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.trunc(data=None, out=None, name=None,
**kwargs)**

   Return the element-wise truncated value of the input.

   The truncated value of the scalar x is the nearest integer i which
   is closer to zero than x is. In short, the fractional part of the
   signed number x is discarded.

   Example:

   ::

      trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]

   The storage type of ``trunc`` output depends upon the input storage
   type:

   ..

      * trunc(default) = default

      * trunc(row_sparse) = row_sparse

   Defined in src/operator/tensor/elemwise_unary_op.cc:L407

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input
        array.

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

**mxnet.ndarray.sparse.zeros_like(data=None, out=None, name=None,
**kwargs)**

   Return an array of zeros with the same shape and type as the input
   array.

   The storage type of ``zeros_like`` output depends on the storage
   type of the input

   * zeros_like(row_sparse) = row_sparse

   * zeros_like(csr) = csr

   * zeros_like(default) = default

   Examples:

   ::

      x = [[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]]

      zeros_like(x) = [[ 0.,  0.,  0.],
                       [ 0.,  0.,  0.]]

   :Parameters:
      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_) -- The input

      * **out** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        **optional*) -- The output NDArray to hold the result.

   :Returns:
      **out** -- The output of this function.

   :Return type:
      `NDArray <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_ or
      list of NDArrays

Sparse NDArray API of MXNet.

**mxnet.ndarray.sparse.zeros(stype, shape, ctx=None, dtype=None,
aux_types=None, **kwargs)**

   Return a new array of given shape and type, filled with zeros.

   :Parameters:
      * **stype** (*string*) -- The storage type of the empty array,
        such as 'row_sparse', 'csr', etc

      * **shape** (*int** or **tuple of int*) -- The shape of the
        empty array

      * **ctx** (*Context**, **optional*) -- An optional device
        context (default is the current default context)

      * **dtype** (*str** or **numpy.dtype**, **optional*) -- An
        optional value type (default is *float32*)

      * **aux_types** (*list of numpy.dtype**, **optional*) -- An
        optional list of types of the aux data for RowSparseNDArray or
        CSRNDArray (default values depends on the storage type)

   :Returns:
      A created array

   :Return type:
      `RowSparseNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_
      or `CSRNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_

   -[ Examples ]-

   >>> mx.nd.sparse.zeros('csr', (1,2))
   <CSRNDArray 1x2 @cpu(0)>
   >>> mx.nd.sparse.zeros('row_sparse', (1,2), ctx=mx.cpu(), dtype='float16').asnumpy()
   array([[ 0.,  0.]], dtype=float16)

**mxnet.ndarray.sparse.empty(stype, shape, ctx=None, dtype=None,
aux_types=None)**

   Returns a new array of given shape and type, without initializing
   entries.

   :Parameters:
      * **stype** (*string*) -- The storage type of the empty array,
        such as 'row_sparse', 'csr', etc

      * **shape** (*int** or **tuple of int*) -- The shape of the
        empty array.

      * **ctx** (*Context**, **optional*) -- An optional device
        context (default is the current default context).

      * **dtype** (*str** or **numpy.dtype**, **optional*) -- An
        optional value type (default is *float32*).

   :Returns:
      A created array.

   :Return type:
      `CSRNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_ or
      `RowSparseNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_

**mxnet.ndarray.sparse.array(source_array, ctx=None, dtype=None,
aux_types=None)**

   Creates a sparse array from any object exposing the array
   interface.

   :Parameters:
      * **source_array** (`RowSparseNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_*,
        *`CSRNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_* or
        **scipy.sparse.csr.csr_matrix*) -- The source sparse array

      * **ctx** (*Context**, **optional*) -- Device context (default
        is the current default context).

      * **dtype** (*str** or **numpy.dtype**, **optional*) -- The data
        type of the output array. The default dtype is
        ``source_array.dtype`` if *source_array* is an *NDArray*,
        *float32* otherwise.

      * **aux_types** (*list of numpy.dtype**, **optional*) -- An
        optional list of types of the aux data for RowSparseNDArray or
        CSRNDArray. The default value for CSRNDArray is [*int64*,
        *int64*] for *indptr* and *indices*. The default value for
        RowSparseNDArray is [*int64*] for *indices*.

   :Returns:
      An array with the same contents as the *source_array*.

   :Return type:
      `RowSparseNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_
      or `CSRNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_

   -[ Examples ]-

   >>> import scipy.sparse as sp
   >>> csr = sp.csr_matrix((2, 100))
   >>> mx.nd.sparse.array(csr)
   <CSRNDArray 2x100 @cpu(0)>
   >>> mx.nd.sparse.array(mx.nd.sparse.zeros('csr', (3, 2)))
   <CSRNDArray 3x2 @cpu(0)>
   >>> mx.nd.sparse.array(mx.nd.sparse.zeros('row_sparse', (3, 2)))
   <RowSparseNDArray 3x2 @cpu(0)>

NDArray API of MXNet.

**mxnet.ndarray.load(fname)**

   Loads an array from file.

   See more details in ``save``.

   :Parameters:
      **fname** (*str*) -- The filename.

   :Returns:
      Loaded data.

   :Return type:
      list of NDArray, `RowSparseNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_
      or `CSRNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_, or
      dict of str to NDArray, `RowSparseNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_
      or `CSRNDArray
      <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_

**mxnet.ndarray.save(fname, data)**

   Saves a list of arrays or a dict of str->array to file.

   Examples of filenames:

   * ``/path/to/file``

   * ``s3://my-bucket/path/to/file`` (if compiled with AWS S3
     supports)

   * ``hdfs://path/to/file`` (if compiled with HDFS supports)

   :Parameters:
      * **fname** (*str*) -- The filename.

      * **data** (`NDArray
        <../ndarray/ndarray.rst#mxnet.ndarray.NDArray>`_*,
        *`RowSparseNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_*
        or *`CSRNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_*,
        or **list of NDArray**, *`RowSparseNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_*
        or *`CSRNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_*,
        or **dict of str to NDArray**, *`RowSparseNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.RowSparseNDArray>`_*
        or *`CSRNDArray
        <../ndarray/sparse.rst#mxnet.ndarray.sparse.CSRNDArray>`_) --
        The data to save.

   -[ Examples ]-

   >>> x = mx.nd.zeros((2,3))
   >>> y = mx.nd.ones((1,4))
   >>> mx.nd.save('my_list', [x,y])
   >>> mx.nd.save('my_dict', {'x':x, 'y':y})
   >>> mx.nd.load('my_list')
   [<NDArray 2x3 @cpu(0)>, <NDArray 1x4 @cpu(0)>]
   >>> mx.nd.load('my_dict')
   {'y': <NDArray 1x4 @cpu(0)>, 'x': <NDArray 2x3 @cpu(0)>}
