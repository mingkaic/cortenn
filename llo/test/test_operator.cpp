
#ifndef DISABLE_OPERATOR_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "exam/exam.hpp"

#include "llo/operator.hpp"


TEST(OPERATOR, Unary)
{
	auto func = [](llo::TensorT<double>& out, const llo::TensorT<double>& in)
	{
		out = in;
	};
	ade::Shape shape({4, 3});
	llo::TensorT<double> out = llo::get_tensor<double>(nullptr, shape);
	std::vector<double> data = {
		34,73,1,67,
		91,91,7,6,
		86,86,85,83,
	};
	llo::DataArg<double> inref_id{
		llo::get_tensorptr(data.data(), shape),
		ade::identity,
		true
	};
	llo::unary<double>(out, inref_id, func);
	{
		double* outptr = out.data();
		std::vector<double> outvec(outptr, outptr + shape.n_elems());
		EXPECT_ARREQ(data, outvec);
	}

	ade::CoordptrT overwrite_mapper(
		new ade::CoordMap([](ade::MatrixT m)
		{
			for (uint8_t i = 0; i < ade::mat_dim; ++i)
			{
				m[i][i] = 1;
			}
			m[1][1] = 0.5;
			m[ade::mat_dim - 1][1] = 0.5;
		}));

	ade::Shape shape2({4, 2});
	llo::DataArg<double> inref_fwd{
		llo::get_tensorptr(data.data(), shape),
		overwrite_mapper,
		true
	};
	std::vector<double> expect_out = {
		34,73,1,67,
		86,86,85,83,
	};
	llo::TensorT<double> out2 = llo::get_tensor<double>(nullptr, shape2);
	llo::unary<double>(out2, inref_fwd, func);
	{
		double* outptr = out2.data();
		std::vector<double> outvec(outptr, outptr + shape2.n_elems());
		EXPECT_ARREQ(expect_out, outvec);
	}

	std::vector<double> data2 = {
		91,91,7,6,
		86,86,85,83,
	};
	std::vector<double> expect_out2 = {
		91,91,7,6,
		86,86,85,83,
		86,86,85,83,
	};
	llo::DataArg<double> inref_bwd{
		llo::get_tensorptr(data2.data(), shape2),
		overwrite_mapper,
		false
	};
	llo::TensorT<double> out3 = llo::get_tensor<double>(nullptr, shape);
	llo::unary<double>(out3, inref_bwd, func);
	{
		double* outptr = out3.data();
		std::vector<double> outvec(outptr, outptr + shape.n_elems());
		EXPECT_ARREQ(expect_out2, outvec);
	}
}


TEST(OPERATOR, Binary)
{
	auto func = [](llo::TensorT<double>& out,
		const llo::TensorT<double>& in, const llo::TensorT<double>& in2)
	{
		out = in - in2;
	};
	ade::Shape shape({4, 3});
	std::vector<double> data = {
		34,73,1,67,
		91,91,7,6,
		86,86,85,83,
	};
	std::vector<double> data2 = {
		75,22,33,86,
		18,99,68,37,
		86,80,47,73,
	};
	ade::Shape reduced_shape({4, 2});
	ade::Shape extended_shape({4, 4});

	ade::CoordptrT overwrite_mapper(
		new ade::CoordMap([](ade::MatrixT m)
		{
			for (uint8_t i = 0; i < ade::mat_dim; ++i)
			{
				m[i][i] = 1;
			}
			m[1][1] = 0.5;
			m[ade::mat_dim - 1][1] = 0.5;
		}));

	// both fwd
	{
		std::vector<double> expect_out = {
			34-75,73-22,1-33,67-86,
			86-86,86-80,85-47,83-73,
		};
		llo::DataArg<double> inref_fwd{
			llo::get_tensorptr(data.data(), shape),
			overwrite_mapper,
			true
		};
		llo::DataArg<double> inref_fwd2{
			llo::get_tensorptr(data2.data(), shape),
			overwrite_mapper,
			true
		};
		llo::TensorT<double> out =
			llo::get_tensor<double>(nullptr, reduced_shape);
		llo::binary<double>(out, inref_fwd, inref_fwd2, func);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + reduced_shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}

	// both bwd
	{
		std::vector<double> expect_out = {
			34-75,73-22,1-33,67-86,
			91-18,91-99,7-68,6-37,
			91-18,91-99,7-68,6-37,
			86-86,86-80,85-47,83-73,
		};
		llo::DataArg<double> inref_bwd{
			llo::get_tensorptr(data.data(), shape),
			overwrite_mapper,
			false
		};
		llo::DataArg<double> inref_bwd2{
			llo::get_tensorptr(data2.data(), shape),
			overwrite_mapper,
			false
		};
		llo::TensorT<double> out =
			llo::get_tensor<double>(nullptr, extended_shape);
		llo::binary<double>(out, inref_bwd, inref_bwd2, func);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + extended_shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}

	std::vector<double> data3 = {
		90,47,47,94,
		18,16,24,23,
		29,16,92,3,
		44,99,71,67,
	};
	std::vector<double> data4 = {
		15,89,96,59,
		2,65,29,89,
	};
	// left fwd, right bwd
	{
		std::vector<double> expect_out = {
			90-15,47-89,47-96,94-59,
			29-2,16-65,92-29,3-89,
			44-2,99-65,71-29,67-89,
		};
		llo::DataArg<double> inref_bwd{
			llo::get_tensorptr(data3.data(), extended_shape),
			overwrite_mapper,
			true
		};
		llo::DataArg<double> inref_bwd2{
			llo::get_tensorptr(data4.data(), reduced_shape),
			overwrite_mapper,
			false
		};
		llo::TensorT<double> out = llo::get_tensor<double>(nullptr, shape);
		llo::binary<double>(out, inref_bwd, inref_bwd2, func);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}
	// left bwd, right fwd
	{
		std::vector<double> expect_out = {
			15-90,89-47,96-47,59-94,
			2-29,65-16,29-92,89-3,
			2-44,65-99,29-71,89-67,
		};
		llo::DataArg<double> inref_bwd{
			llo::get_tensorptr(data4.data(), reduced_shape),
			overwrite_mapper,
			false
		};
		llo::DataArg<double> inref_bwd2{
			llo::get_tensorptr(data3.data(), extended_shape),
			overwrite_mapper,
			true
		};
		llo::TensorT<double> out = llo::get_tensor<double>(nullptr, shape);
		llo::binary<double>(out, inref_bwd, inref_bwd2, func);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}
}


TEST(OPERATOR, Nnary)
{
	auto func = [](double& acc, const double& in)
	{
		acc += in;
	};
	auto tensfunc = [](llo::TensorT<double>& acc, const llo::TensorT<double>& in)
	{
		acc += in;
	};
	ade::Shape shape({4, 3});
	std::vector<double> data = {
		34,73,1,67,
		91,91,7,6,
		86,86,85,83,
	};
	std::vector<double> data2 = {
		75,22,33,86,
		18,99,68,37,
		86,80,47,73,
	};
	ade::Shape reduced_shape({4, 2});
	ade::Shape extended_shape({4, 4});

	ade::CoordptrT overwrite_mapper(
		new ade::CoordMap([](ade::MatrixT m)
		{
			for (uint8_t i = 0; i < ade::mat_dim; ++i)
			{
				m[i][i] = 1;
			}
			m[1][1] = 0.5;
			m[ade::mat_dim - 1][1] = 0.5;
		}));

	// both fwd
	{
		std::vector<double> expect_out = {
			34+75,73+22,1+33,67+86,
			91+18+86+86,91+99+86+80,7+68+85+47,6+37+83+73,
		};
		llo::DataArg<double> inref_fwd{
			llo::get_tensorptr(data.data(), shape),
			overwrite_mapper,
			true
		};
		llo::DataArg<double> inref_fwd2{
			llo::get_tensorptr(data2.data(), shape),
			overwrite_mapper,
			true
		};
		llo::TensorT<double> out =
			llo::get_tensor<double>(nullptr, reduced_shape);
		llo::nnary<double>(out, {inref_fwd, inref_fwd2}, func, tensfunc);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + reduced_shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}

	// both bwd
	{
		std::vector<double> expect_out = {
			34+75,73+22,1+33,67+86,
			91+18,91+99,7+68,6+37,
			91+18,91+99,7+68,6+37,
			86+86,86+80,85+47,83+73,
		};
		llo::DataArg<double> inref_bwd{
			llo::get_tensorptr(data.data(), shape),
			overwrite_mapper,
			false
		};
		llo::DataArg<double> inref_bwd2{
			llo::get_tensorptr(data2.data(), shape),
			overwrite_mapper,
			false
		};
		llo::TensorT<double> out =
			llo::get_tensor<double>(nullptr, extended_shape);
		llo::nnary<double>(out, {inref_bwd, inref_bwd2}, func, tensfunc);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + extended_shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}

	std::vector<double> data3 = {
		90,47,47,94,
		18,16,24,23,
		29,16,92,3,
		44,99,71,67,
	};
	std::vector<double> data4 = {
		15,89,96,59,
		2,65,29,89,
	};
	std::vector<double> expect_out = {
		90+15,47+89,47+96,94+59,
		18+29+2,16+16+65,24+92+29,23+3+89,
		44+2,99+65,71+29,67+89,
	};
	// fwd, then bwd
	{
		llo::DataArg<double> inref_bwd{
			llo::get_tensorptr(data3.data(), extended_shape),
			overwrite_mapper,
			true
		};
		llo::DataArg<double> inref_bwd2{
			llo::get_tensorptr(data4.data(), reduced_shape),
			overwrite_mapper,
			false
		};
		llo::TensorT<double> out = llo::get_tensor<double>(nullptr, shape);
		llo::nnary<double>(out, {inref_bwd, inref_bwd2}, func, tensfunc);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}
	// bwd, then fwd
	{
		llo::DataArg<double> inref_bwd{
			llo::get_tensorptr(data4.data(), reduced_shape),
			overwrite_mapper,
			false
		};
		llo::DataArg<double> inref_bwd2{
			llo::get_tensorptr(data3.data(), extended_shape),
			overwrite_mapper,
			true
		};
		llo::TensorT<double> out = llo::get_tensor<double>(nullptr, shape);
		llo::nnary<double>(out, {inref_bwd, inref_bwd2}, func, tensfunc);
		{
			double* outptr = out.data();
			std::vector<double> outvec(outptr, outptr + shape.n_elems());
			EXPECT_ARREQ(expect_out, outvec);
		}
	}
}


#endif // DISABLE_OPERATOR_TEST
