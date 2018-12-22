import unittest
import numpy as np
import tensorflow as tf

import llo.age as age
import llo.llo as llo

class LLOTest(unittest.TestCase):
    def _common_unary(self, shape, api, real, derive):
        data = np.random.rand(*shape) * 234
        var = llo.variable(data, 'var')
        out = api(var)

        fout = llo.evaluate(out, dtype=np.dtype(float))
        self.assertTrue(np.array_equal(real(data), fout))

        var2 = llo.variable(data, 'var2')
        ex = llo.derive(out, var)
        zero = llo.derive(out, var2)

        data0 = np.zeros(shape, dtype=float)
        der = llo.evaluate(ex)
        rej = llo.evaluate(zero)
        exdata = derive(data)
        self.assertTrue(np.allclose(exdata, der))
        self.assertEqual(exdata.shape, der.shape)
        self.assertTrue(np.array_equal(data0, rej))

    def _common_binary(self, shape, api, real, derive):
        data = np.random.rand(*shape)
        data2 = np.random.rand(*shape)
        var = llo.variable(data, 'var')
        var2 = llo.variable(data2, 'var2')
        out = api(var, var2)
        both = api(var, var)

        fout = llo.evaluate(out, dtype=np.dtype(float))
        fboth = llo.evaluate(both, dtype=np.dtype(float))
        self.assertTrue(np.array_equal(real(data, data2), fout))
        self.assertTrue(np.array_equal(real(data, data), fboth))

        var3 = llo.variable(data, 'var3')

        zero = llo.derive(out, var3)
        ex = llo.derive(out, var)
        ex2 = llo.derive(out, var2)
        ex3 = llo.derive(both, var)

        rej = llo.evaluate(zero)
        der = llo.evaluate(ex)
        der2 = llo.evaluate(ex2)
        der3 = llo.evaluate(ex3)

        data0 = np.zeros(shape, dtype=float)
        exdata = derive(0, (data, data2))
        exdata2 = derive(1, (data, data2))
        exdata3 = derive(0, (data, data)) + derive(1, (data, data))

        self.assertTrue(np.array_equal(data0, rej))

        self.assertTrue(np.allclose(exdata, der))
        self.assertEqual(exdata.shape, der.shape)

        self.assertTrue(np.allclose(exdata2, der2))
        self.assertEqual(exdata2.shape, der2.shape)

        self.assertTrue(np.allclose(exdata3, der3))
        self.assertEqual(exdata3.shape, der3.shape)

    def _common_tfbinary(self, shape, api, real):
        data = np.random.rand(*shape)
        data2 = np.random.rand(*shape)
        var = llo.variable(data, 'var')
        var2 = llo.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        sess = tf.Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)

        out = api(var, var2)
        both = api(var, var)
        tf_out = real(tf_var, tf_var2)
        tf_both = real(tf_var, tf_var)

        fout = llo.evaluate(out, dtype=np.dtype(float))
        fboth = llo.evaluate(both, dtype=np.dtype(float))
        tf_fout = sess.run(tf_out)
        tf_fboth = sess.run(tf_both)

        self.assertTrue(np.allclose(tf_fout, fout))
        self.assertEqual(tf_fout.shape, fout.shape)

        self.assertTrue(np.allclose(tf_fboth, fboth))
        self.assertEqual(tf_fboth.shape, fboth.shape)

        var3 = llo.variable(data, 'var3')

        zero = llo.derive(out, var3)
        ex = llo.derive(out, var)
        ex2 = llo.derive(out, var2)
        ex3 = llo.derive(both, var)

        rej = llo.evaluate(zero)
        der = llo.evaluate(ex)
        der2 = llo.evaluate(ex2)
        der3 = llo.evaluate(ex3)

        data0 = np.zeros(shape, dtype=float)

        tf_grad, tf_grad2 = tf.gradients(tf_out, [tf_var, tf_var2])
        tf_grad3 = tf.gradients(tf_both, [tf_var])[0]

        exdata = sess.run(tf_grad)
        exdata2 = sess.run(tf_grad2)
        exdata3 = sess.run(tf_grad3)

        self.assertTrue(np.array_equal(data0, rej))

        self.assertTrue(np.allclose(exdata, der))
        self.assertEqual(exdata.shape, der.shape)

        self.assertTrue(np.allclose(exdata2, der2))
        self.assertEqual(exdata2.shape, der2.shape)

        self.assertTrue(np.allclose(exdata3, der3))
        self.assertEqual(exdata3.shape, der3.shape)

    def test_variable(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        data0 = np.zeros(shape, dtype=float)
        data = np.random.rand(3, 4, 5) * 234
        var = llo.variable(data, 'var')
        fout = llo.evaluate(var, dtype=np.dtype(float))
        iout = llo.evaluate(var, dtype=np.dtype(int))

        self.assertEqual(tuple(shape), fout.shape)
        self.assertEqual(tuple(shape), iout.shape)
        self.assertTrue(np.array_equal(data, fout))
        self.assertTrue(np.array_equal(data.astype(int), iout))

        var2 = llo.variable(data, 'var2')
        one = llo.derive(var, var)
        zero = llo.derive(var, var2)

        out1 = llo.evaluate(one)
        out0 = llo.evaluate(zero)
        self.assertEqual(tuple(shape), out1.shape)
        self.assertEqual(tuple(shape), out0.shape)
        self.assertTrue(np.array_equal(data1, out1))
        self.assertTrue(np.array_equal(data0, out0))

    def test_abs(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.abs, abs,
            lambda data: data / abs(data))

    def test_neg(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        self._common_unary(shape, age.neg, lambda a: -a,
            lambda data: -data1)

    def test_sin(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.sin, np.sin, np.cos)

    def test_cos(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.cos, np.cos, lambda x: -np.sin(x))

    def test_tan(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.tan, np.tan,
            lambda x: (1.0 / np.cos(x)) / np.cos(x))

    def test_exp(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.exp, np.exp, np.exp)

    def test_log(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.log, np.log, lambda x: 1.0 / x)

    def test_sqrt(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.sqrt, np.sqrt,
            lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def test_round(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        self._common_unary(shape, age.round, np.round, lambda x: data1)

    # def test_flip(self):
    #     pass

    def test_pow(self):
        shape = [3, 4, 5]
        def pow_der(i, data):
            a, b = data
            if i == 0:
                return b * a ** (b - 1)
            return a ** b * np.log(a)
        self._common_binary(shape, age.pow, lambda x, y: x ** y, pow_der)

    def test_add(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        self._common_binary(shape, age.add, lambda x, y: x + y,
            lambda i, data: data1)

    def test_sub(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        def sub_der(i, data):
            if i == 0:
                return data1
            return -data1
        self._common_binary(shape, age.sub, lambda x, y: x - y, sub_der)

    def test_mul(self):
        shape = [3, 4, 5]
        def mul_der(i, data):
            if i == 0:
                return data[1]
            return data[0]
        self._common_binary(shape, age.mul, lambda x, y: x * y, mul_der)

    def test_div(self):
        shape = [3, 4, 5]
        def div_der(i, data):
            a, b = data
            if i == 0:
                return 1 / b
            return -a / (b * b)
        self._common_binary(shape, age.div, lambda x, y: x / y, div_der)

    def test_matmul(self):
        shape = [5, 5]
        self._common_tfbinary(shape, age.matmul, tf.matmul)

if __name__ == "__main__":
    unittest.main()
