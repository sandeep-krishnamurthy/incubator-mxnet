import mxnet as mx
import mxnet.ndarray as nd
import time

"""
Resources:
1. https://mxnet.incubator.apache.org/versions/master/faq/new_op.html
2. https://github.com/apache/incubator-mxnet/pull/5589/files
3. See tests/python/test_operator.py
"""

# Custom Op for Elementwise Matrix Multiplication
class MMul(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0]*in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], in_data[1])
        self.assign(in_grad[1], req[1], in_data[0])

@mx.operator.register("mmul")
class MulProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MulProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['lhs', 'rhs']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return MMul()

"""
# Custom NDArray Op for Elementwise Matrix Multiplication
class MMulND(mx.operator.NDArrayOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0]*in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], in_data[1])
        self.assign(in_grad[1], req[1], in_data[0])

@mx.operator.register("mmulnd")
class MultProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MultProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['lhs', 'rhs']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return MMulND()
"""

# Gluon Block for Elementwise Matrix Multiplication
class MMulBlock(mx.gluon.Block):
    def forward(self, lhs, rhs):
        # Invoke elementwise NDArray matrix multiplication
        return lhs * rhs

# Do Benchmarks

def run_benchmarks(in_shape=(1000, 1000)):
    """
        Returns benchmarks results in the format:
        (custom_op_time, native_ndarray_op_time, gluon_block_time)
    """
    # Inputs
    in_shape = (10000, 10000)

    # 1. Custom Op
    lhs = mx.nd.full(shape=in_shape, val=2)
    rhs = mx.nd.ones(shape=in_shape)
    lhs.attach_grad()
    rhs.attach_grad()
    nd.waitall()

    start = time.time()
    with mx.autograd.record():
        res = mx.nd.Custom(lhs, rhs, name="mmul", op_type="mmul")
    res.backward()
    nd.waitall()
    custom_op_time = time.time() - start

    """
    print("Custom Op Results")
    print("Result - ", res)
    print("Grad LHS - ", lhs.grad)
    print("Grad RHS - ", rhs.grad)
    """

    # 2. Native Op
    lhs = mx.nd.full(shape=in_shape, val=2)
    rhs = mx.nd.ones(shape=in_shape)
    lhs.attach_grad()
    rhs.attach_grad()
    nd.waitall()

    start = time.time()
    with mx.autograd.record():
        res = lhs * rhs
    res.backward()
    nd.waitall()
    native_nd_op_time = time.time() - start

    """
    print("Native NDArray Op Results")
    print("Result - ", res)
    print("Grad LHS - ", lhs.grad)
    print("Grad RHS - ", rhs.grad)
    """

    # 3. Gluon Block Op
    lhs = mx.nd.full(shape=in_shape, val=2)
    rhs = mx.nd.ones(shape=in_shape)
    lhs.attach_grad()
    rhs.attach_grad()
    nd.waitall()

    start = time.time()
    m_mul_block = MMulBlock()
    with mx.autograd.record():
        res = m_mul_block(lhs, rhs)
    res.backward()
    nd.waitall()
    gluon_block_native_op_time = time.time() - start

    """
    print("Gluon Block with Native NDArray Op Results")
    print("Result - ", res)
    print("Grad LHS - ", lhs.grad)
    print("Grad RHS - ", rhs.grad)
    """
    return custom_op_time, native_nd_op_time, gluon_block_native_op_time

# Run Benchmarks
runs = 100
in_shape=(10000, 1)
custom_op_time = native_nd_op_time = gluon_block_native_op_time = 0

for _ in range(runs):
    cur_custom_op_time, cur_native_nd_op_time, cur_gluon_block_native_op_time = run_benchmarks(in_shape)
    custom_op_time += cur_custom_op_time
    native_nd_op_time += cur_native_nd_op_time
    gluon_block_native_op_time += cur_gluon_block_native_op_time

print("Benchmark Results")
print("Operation - Elementwise matrix multiplication of NDArray with shape - ", in_shape)
print("Custom Op - %.5f sec"%(custom_op_time/runs))
print("Native ND Op - %.5f sec"%(native_nd_op_time/runs))
print("Gluon Block with Native ND Op - %.5f sec"%(gluon_block_native_op_time/runs))

"""
April 10, 2019 Results on a Mac:
Averaged over 100 runs:

Custom Op - 2.43820 sec
Native ND Op - 2.56218 sec
Gluon Block with Native ND Op - 2.60773 sec

"""