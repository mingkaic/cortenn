
#ifndef DISABLE_OPERATOR_TEST


#include "gtest/gtest.h"

#include "llo/test/common.hpp"

#include "llo/operator.hpp"


TEST(OPERATOR, Unary)
{
	auto func = [](const double& in) -> double
	{
		return in;
	};
	ade::Shape shape({4, 3});
	std::vector<double> out(12);
	std::vector<double> data = {
		34,73,1,67,
		91,91,7,6,
		86,86,85,83,
	};
	llo::VecRef<double> inref_id{&data[0], shape, ade::identity, true};
	llo::unary<double>(&out[0], shape, inref_id, func);
	EXPECT_ARREQ(data, out);

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
	llo::VecRef<double> inref_fwd{&data[0], shape, overwrite_mapper, true};
	std::vector<double> expect_out = {
		34,73,1,67,
		86,86,85,83,
	};
	std::vector<double> out2(8);
	llo::unary<double>(&out2[0], shape2, inref_fwd, func);
	EXPECT_ARREQ(expect_out, out2);

	std::vector<double> data2 = {
		91,91,7,6,
		86,86,85,83,
	};
	std::vector<double> expect_out2 = {
		91,91,7,6,
		86,86,85,83,
		86,86,85,83,
	};
	llo::VecRef<double> inref_bwd{&data2[0], shape2, overwrite_mapper, false};
	std::vector<double> out3(12);
	llo::unary<double>(&out3[0], shape, inref_bwd, func);
	EXPECT_ARREQ(expect_out2, out3);
}


TEST(OPERATOR, Binary)
{
	auto func = [](const double& in, const double& in2) -> double
	{
		return in - in2;
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
		llo::VecRef<double> inref_fwd{&data[0], shape, overwrite_mapper, true};
		llo::VecRef<double> inref_fwd2{&data2[0], shape, overwrite_mapper, true};
		std::vector<double> out(8);
		llo::binary<double,double,double>(&out[0], reduced_shape,
			inref_fwd, inref_fwd2, func);
		EXPECT_ARREQ(expect_out, out);
	}

	// both bwd
	{
		std::vector<double> expect_out = {
			34-75,73-22,1-33,67-86,
			91-18,91-99,7-68,6-37,
			91-18,91-99,7-68,6-37,
			86-86,86-80,85-47,83-73,
		};
		llo::VecRef<double> inref_bwd{&data[0], shape, overwrite_mapper, false};
		llo::VecRef<double> inref_bwd2{&data2[0], shape, overwrite_mapper, false};
		std::vector<double> out(16);
		llo::binary<double,double,double>(&out[0], extended_shape,
			inref_bwd, inref_bwd2, func);
		EXPECT_ARREQ(expect_out, out);
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
		llo::VecRef<double> inref_bwd{&data3[0], extended_shape, overwrite_mapper, true};
		llo::VecRef<double> inref_bwd2{&data4[0], reduced_shape, overwrite_mapper, false};
		std::vector<double> out(12);
		llo::binary<double,double,double>(&out[0], shape,
			inref_bwd, inref_bwd2, func);
		EXPECT_ARREQ(expect_out, out);
	}
	// left bwd, right fwd
	{
		std::vector<double> expect_out = {
			15-90,89-47,96-47,59-94,
			2-29,65-16,29-92,89-3,
			2-44,65-99,29-71,89-67,
		};
		llo::VecRef<double> inref_bwd{&data4[0], reduced_shape, overwrite_mapper, false};
		llo::VecRef<double> inref_bwd2{&data3[0], extended_shape, overwrite_mapper, true};
		std::vector<double> out(12);
		llo::binary<double,double,double>(&out[0], shape,
			inref_bwd, inref_bwd2, func);
		EXPECT_ARREQ(expect_out, out);
	}
}


TEST(OPERATOR, Nnary)
{
	auto func = [](double& acc, const double& in)
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
		llo::VecRef<double> inref_fwd{&data[0], shape, overwrite_mapper, true};
		llo::VecRef<double> inref_fwd2{&data2[0], shape, overwrite_mapper, true};
		std::vector<double> out(8);
		llo::nnary<double>(&out[0], reduced_shape,
			{inref_fwd, inref_fwd2}, func);
		EXPECT_ARREQ(expect_out, out);
	}

	// both bwd
	{
		std::vector<double> expect_out = {
			34+75,73+22,1+33,67+86,
			91+18,91+99,7+68,6+37,
			91+18,91+99,7+68,6+37,
			86+86,86+80,85+47,83+73,
		};
		llo::VecRef<double> inref_bwd{&data[0], shape, overwrite_mapper, false};
		llo::VecRef<double> inref_bwd2{&data2[0], shape, overwrite_mapper, false};
		std::vector<double> out(16);
		llo::nnary<double>(&out[0], extended_shape,
			{inref_bwd, inref_bwd2}, func);
		EXPECT_ARREQ(expect_out, out);
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
		llo::VecRef<double> inref_bwd{&data3[0], extended_shape, overwrite_mapper, true};
		llo::VecRef<double> inref_bwd2{&data4[0], reduced_shape, overwrite_mapper, false};
		std::vector<double> out(12);
		llo::nnary<double>(&out[0], shape,
			{inref_bwd, inref_bwd2}, func);
		EXPECT_ARREQ(expect_out, out);
	}
	// bwd, then fwd
	{
		llo::VecRef<double> inref_bwd{&data4[0], reduced_shape, overwrite_mapper, false};
		llo::VecRef<double> inref_bwd2{&data3[0], extended_shape, overwrite_mapper, true};
		std::vector<double> out(12);
		llo::nnary<double>(&out[0], shape,
			{inref_bwd, inref_bwd2}, func);
		EXPECT_ARREQ(expect_out, out);
	}
}


#endif // DISABLE_OPERATOR_TEST
