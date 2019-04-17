import time
import mxnet as mx
from mxnet import ndarray as nd

"""
This is a helper benchmark script to understand overhead of 
custom operator in MXNet.

It does a simple elementwise addition to make sure computation
is not too much and we can see custom operator logistics overhead.

1. Tests Custom v/s Imperative (Native NDArray)
2. Tests Custom v/s Symbolic (Native Symbol with Simple Bind)
"""

# 1. Define Custom Operator - Elementwise Matrix Multiplication
class CustomAddOne(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0] + 1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("CustomAddOne")
class CustomAddOneProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CustomAddOneProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['in']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        # inputs, outputs, aux
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return CustomAddOne()

# Benchmark will be done on the following operation:
# native_add -> native_add -> native_add -> CUSTOM_ADD -> native_add -> native_add -> native_add
# Versus
# native_add -> native_add -> native_add -> NATIVE_ADD -> native_add -> native_add -> native_add

def run_forward_only_benchmark():
    ##### NATIVE BENCHMARKS ######
    native_total_time = 0
    for _ in range(runs):
        inp = mx.nd.ones(shape=shape)
        nd.waitall()

        native_start_time = time.time()

        # Forward Only
        res1 = inp + 1
        res2 = res1 + 1
        res3 = res2 + 1
        res4 = res3 + 1
        res5 = res4 + 1
        res6 = res5 + 1
        res7 = res6 + 1

        nd.waitall()
        native_end_time = time.time()
        native_total_time += (native_end_time - native_start_time)

    print("Average Native operation forward time - ", native_total_time/runs)

    ######## CUSTOM BENCHMARKS ########
    custom_total_time = 0
    for _ in range(runs):
        inp = mx.nd.ones(shape=shape)
        nd.waitall()

        custom_start_time = time.time()

        # Forward Only
        res1 = inp + 1
        res2 = res1 + 1
        res3 = res2 + 1
        res4 = nd.Custom(res3, name="customaddone", op_type="CustomAddOne")
        res5 = res4 + 1
        res6 = res5 + 1
        res7 = res6 + 1

        nd.waitall()
        custom_end_time = time.time()
        custom_total_time += (custom_end_time - custom_start_time)

    print("Average Custom operation Forward time - ", custom_total_time/runs)

def run_forward_backward_benchmark():
        ##### NATIVE BENCHMARKS ######
    native_total_time = 0
    for _ in range(runs):
        inp = mx.nd.ones(shape=shape)
        inp.attach_grad()
        nd.waitall()

        native_start_time = time.time()
        with mx.autograd.record():
            # Forward
            res1 = inp + 1
            res2 = res1 + 1
            res3 = res2 + 1
            res4 = res3 + 1
            res5 = res4 + 1
            res6 = res5 + 1
            res7 = res6 + 1
        # Backward
        res7.backward()
        nd.waitall()

        native_end_time = time.time()
        native_total_time += (native_end_time - native_start_time)

    print("Average Native operation forward+backward time - ", native_total_time/runs)

    ######## CUSTOM BENCHMARKS ########
    custom_total_time = 0
    for _ in range(runs):
        inp = mx.nd.ones(shape=shape)
        inp.attach_grad()
        nd.waitall()

        custom_start_time = time.time()
        with mx.autograd.record():
            # Forward
            res1 = inp + 1
            res2 = res1 + 1
            res3 = res2 + 1
            res4 = nd.Custom(res3, name="customaddone", op_type="CustomAddOne")
            res5 = res4 + 1
            res6 = res5 + 1
            res7 = res6 + 1
        # Backward
        res7.backward()
        nd.waitall()

        custom_end_time = time.time()
        custom_total_time += (custom_end_time - custom_start_time)

    print("Average Custom operation Forward+Backward time - ", custom_total_time/runs)

# Parameters
shape = (100, 1)
runs = 100

# Run Benchmarks
run_forward_only_benchmark()
run_forward_backward_benchmark()

"""
RESULTS:

####### ON MAC #######
Default Engine

Average Native operation forward time -  0.00037915945053100585
Average Custom operation Forward time -  0.0008458781242370605
Average Native operation forward+backward time -  0.0010153484344482423
Average Custom operation Forward+Backward time -  0.0016492009162902832

Naive Engine

[17:31:14] src/engine/engine.cc:55: MXNet start using engine: NaiveEngine
Average Native operation forward time -  0.00031911611557006837
Average Custom operation Forward time -  0.000781712532043457
Average Native operation forward+backward time -  0.0008284425735473632
Average Custom operation Forward+Backward time -  0.0013871908187866211

##### ON P3.8X (32 CORES) CPU MODE #######
Default Engine
('Average Native operation forward time - ', 0.0003200197219848633)
('Average Custom operation Forward time - ', 0.000811471939086914)
('Average Native operation forward+backward time - ', 0.0008977103233337402)
('Average Custom operation Forward+Backward time - ', 0.0014853644371032714)

Naive Engine
[00:35:25] src/engine/engine.cc:55: MXNet start using engine: NaiveEngine
('Average Native operation forward time - ', 0.00021964073181152344)
('Average Custom operation Forward time - ', 0.0006221365928649902)
('Average Native operation forward+backward time - ', 0.0004655289649963379)
('Average Custom operation Forward+Backward time - ', 0.0010217475891113282)
"""