#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pbm.graph_pb2 as pb

graph = pb.Graph()
with open("/tmp/gd_test.pbx", "rb") as f:
    graph.ParseFromString(f.read())

sources = list(map(lambda node: node.source,
    filter(lambda node: node.WhichOneof("detail") == "source",\
    graph.nodes)))

eout = sources[0]
tin = sources[5]
# print(eout.label)
# print(tin.label)
shp_eout = [ord(c) for c in eout.shape]
shp_tin = [ord(c) for c in tin.shape]
# print(shp_eout, shp_tin)
out_feed = np.array(eout.double_arrs.data).reshape([3, 5])
train_feed = np.array(tin.double_arrs.data).reshape([3, 10])

v3 = sources[6]
v2 = sources[7]
v1 = sources[8]
v0 = sources[9]
# print(v3.label)
# print(v2.label)
# print(v1.label)
# print(v0.label)
shp3 = [ord(c) for c in v3.shape]
shp2 = [ord(c) for c in v2.shape]
shp1 = [ord(c) for c in v1.shape]
shp0 = [ord(c) for c in v0.shape]
# print(shp3, shp2, shp1, shp0)
data3 = np.array(v3.double_arrs.data).reshape([10, 9])
data2 = np.array(v2.double_arrs.data).reshape([9])
data1 = np.array(v1.double_arrs.data).reshape([9, 5])
data0 = np.array(v0.double_arrs.data).reshape([5])

train_in = tf.placeholder(tf.float64, shape=[3, 10])
expected_out = tf.placeholder(tf.float64, shape=[3, 5])
var3 = tf.Variable(data3)
var2 = tf.Variable(data2)
var1 = tf.Variable(data1)
var0 = tf.Variable(data0)

out1 = tf.add(tf.matmul(train_in, var3), var2)
sig1 = 1 / (1 + tf.exp(-out1))
out2 = tf.add(tf.matmul(sig1, var1), var0)
sig2 = 1 / (1 + tf.exp(-out2))

ierr = tf.subtract(expected_out, sig2)
err = tf.pow(tf.subtract(expected_out, sig2), 2)

gvar3, gvar2, gvar1, gvar0 = tf.gradients(err, [var3, var2, var1, var0])

sess = tf.Session()
sess.run(var3.initializer)
sess.run(var2.initializer)
sess.run(var1.initializer)
sess.run(var0.initializer)
errout = sess.run(err, feed_dict={
    train_in: train_feed,
    expected_out: out_feed
})
gvar3out = sess.run(gvar3, feed_dict={
    train_in: train_feed,
    expected_out: out_feed
})
gvar2out = sess.run(gvar2, feed_dict={
    train_in: train_feed,
    expected_out: out_feed
})
gvar1out = sess.run(gvar1, feed_dict={
    train_in: train_feed,
    expected_out: out_feed
})
gvar0out = sess.run(gvar0, feed_dict={
    train_in: train_feed,
    expected_out: out_feed
})
print(gvar3out)
print(gvar2out)
print(gvar1out)
print(gvar0out)
print(errout)
