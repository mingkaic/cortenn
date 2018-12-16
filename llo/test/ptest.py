import unittest
import numpy as np

import llo.age as age
import llo.llo as llo

class LLOTest(unittest.TestCase):
    def _common_api(self, shape, api, real, derive):
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
        self._common_api(shape, age.abs, abs,
            lambda data: data / abs(data))

    def test_neg(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        self._common_api(shape, age.neg, lambda a: -a,
            lambda data: -data1)

    def test_sin(self):
        shape = [3, 4, 5]
        self._common_api(shape, age.sin, np.sin, np.cos)

    def test_cos(self):
        shape = [3, 4, 5]
        self._common_api(shape, age.cos, np.cos, lambda x: -np.sin(x))

    def test_tan(self):
        shape = [3, 4, 5]
        self._common_api(shape, age.tan, np.tan,
            lambda x: (1.0 / np.cos(x)) / np.cos(x))

    def test_exp(self):
        shape = [3, 4, 5]
        self._common_api(shape, age.exp, np.exp, np.exp)

    def test_log(self):
        shape = [3, 4, 5]
        self._common_api(shape, age.log, np.log, lambda x: 1.0 / x)

    def test_sqrt(self):
        shape = [3, 4, 5]
        self._common_api(shape, age.sqrt, np.sqrt,
            lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def test_round(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=float)
        self._common_api(shape, age.round, np.round, lambda x: data1)

if __name__ == "__main__":
    unittest.main()
