''' Generate test cases with tensorflow operations '''

import math
import tensorflow as tf
import argparse
import numpy as np
import functools

import retrop.generate as gen
import retrop.client as client

# N_LIMIT = 32000
N_LIMIT = 16000
RANK_LIMIT = 6

def make_shape(io, nrank=RANK_LIMIT):
    rank = io.get_arr("rank", int, 1, (1, nrank))[0]
    dimlimit = min(10 ** (math.log(N_LIMIT, 10) / rank), 255)
    shape = io.get_arr("shape", int, rank, (1, int(dimlimit)))
    n = functools.reduce(lambda d1, d2 : d1 * d2, shape)
    return shape, n, dimlimit

def unary(name, func, pos=False):
    io = gen.GenIO(name)
    shape, n, _ = make_shape(io)
    if pos:
        data = io.get_arr("data", float, n, (0.1, 1.0))
    else:
        data = io.get_arr("data", float, n, (-1.0, 1.0))
    shaped_data = np.reshape(data, shape)

    var = tf.Variable(shaped_data)
    out = func(var)
    ga = tf.gradients(out, [var])[0]

    with tf.Session() as sess:
        sess.run(var.initializer)
        outdata = sess.run(out)
        io.set_arr("unary_out", list(np.reshape(outdata, [n])), float)

        outga = sess.run(ga)
        io.set_arr("unary_ga", list(np.reshape(outga, [n])), float)

        io.send()

def binary(name, func, arange=(-1.0, 1.0), brange=(-1.0, 1.0)):
    io = gen.GenIO(name)
    shape, n, _ = make_shape(io)
    data = io.get_arr("data", float, n, arange)
    data2 = io.get_arr("data2", float, n, brange)
    shaped_data = np.reshape(data, shape)
    shaped_data2 = np.reshape(data2, shape)

    var = tf.Variable(shaped_data)
    var2 = tf.Variable(shaped_data2)
    out = func(var, var2)
    ga, gb = tf.gradients(out, [var, var2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outdata = sess.run(out)
        io.set_arr("binary_out", list(np.reshape(outdata, [n])), float)

        outga = sess.run(ga)
        io.set_arr("binary_ga", list(np.reshape(outga, [n])), float)

        outgb = sess.run(gb)
        io.set_arr("binary_gb", list(np.reshape(outgb, [n])), float)

        io.send()

def matmul():
    label = 'REGRESS::Matmul'
    print('generating ' + label)
    io = gen.GenIO(label)
    dimlimit = math.sqrt(N_LIMIT)
    ashape = io.get_arr("ashape", int, 2, (1, dimlimit))
    bdim = io.get_arr("bdim", int, 1, (1, dimlimit))[0]
    bshape = [bdim, ashape[0]]
    an = ashape[0] * ashape[1]
    bn = bdim * ashape[0]
    data = io.get_arr("data", float, an, (-1.0, 1.0))
    data2 = io.get_arr("data2", float, bn, (-1.0, 1.0))
    shaped_data = np.reshape(data, ashape[::-1])
    shaped_data2 = np.reshape(data2, bshape[::-1])

    var = tf.Variable(shaped_data)
    var2 = tf.Variable(shaped_data2)
    out = tf.matmul(var, var2)
    ga, gb = tf.gradients(out, [var, var2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outdata = sess.run(out)
        nout = outdata.shape[0] * outdata.shape[1]
        io.set_arr("matmul_out", list(np.reshape(outdata, [nout])), float)

        outga = sess.run(ga)
        io.set_arr("matmul_ga", list(np.reshape(outga, [an])), float)

        outgb = sess.run(gb)
        io.set_arr("matmul_gb", list(np.reshape(outgb, [bn])), float)

        io.send()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate tensorflow testdata')
    parser.add_argument('--cert', type=str, dest='cert', default='certs/server.crt',
        help="Path to dora server's x509 certificate file")
    parser.add_argument('--address', type=str, dest='addr', default='localhost:10000',
        help='Dora server address')

    args = parser.parse_args()
    cert = open(args.cert).read()

    client.init(args.addr, cert)
    print("client initialized")

    ufs = [
        ('REGRESS::Abs', tf.abs, False),
        ('REGRESS::Neg', tf.negative, False),
        ('REGRESS::Sin', tf.sin, False),
        ('REGRESS::Cos', tf.cos, False),
        ('REGRESS::Tan', tf.tan, False),
        ('REGRESS::Exp', tf.exp, False),
        ('REGRESS::Log', tf.log, True),
        ('REGRESS::Sqrt', tf.sqrt, True)
    ]

    for name, f, pos in ufs:
        print("generating " + name)
        unary(name, f, pos=pos)

    bfs = [
        ('REGRESS::Pow', tf.pow, ((1.0, 6.0), (-1.0, 1.0))),
        ('REGRESS::Add', tf.add, None),
        ('REGRESS::Sub', tf.subtract, None),
        ('REGRESS::Mul', tf.multiply, None),
        ('REGRESS::Div', tf.div, ((-1.0, 1.0), (0.1, 1.0)))
    ]

    for name, f, bounds in bfs:
        print('generating ' + name)
        if bounds is not None:
            binary(name, f, arange = bounds[0], brange = bounds[1])
        else:
            binary(name, f)

    matmul()
