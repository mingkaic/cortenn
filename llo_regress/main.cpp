#include <cassert>

#include "gtest/gtest.h"

#include "anteroc/testcase.hpp"

#include "llo/generated/api.hpp"
#include "llo/data.hpp"
#include "llo/eval.hpp"
#include "llo/zprune.hpp"


void EXPECT_DATA_EQ (std::string name, std::vector<double> expect, std::vector<double> got)
{
	ASSERT_EQ(expect.size(), got.size());
	for (size_t i = 0, n = expect.size(); i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(expect[i], got[i]) << " @name=" << name << ",index=" << i;
	}
}


void EXPECT_DATA_APPROX (std::string name, std::vector<double> expect, std::vector<double> got)
{
	ASSERT_EQ(expect.size(), got.size());
	for (size_t i = 0, n = expect.size(); i < n; ++i)
	{
		double err = expect[i] - got[i];
		EXPECT_GT(0.000000001, err) << " @name=" << name << ",index=" << i;
	}
}


static ade::Shape get_shape (testify::GeneratedCase& gcase)
{
	auto& inputs = gcase.inputs();
	auto it = inputs.find("shape");
	if (inputs.end() == it)
	{
		throw std::runtime_error("shape not found");
	}
	assert(it->second.has_dint64s());
	const testify::Int64s& arr = it->second.dint64s();
	auto temp = arr.data();
	std::vector<ade::DimT> slist(temp.begin(), temp.end());
	return ade::Shape(slist);
}


static std::vector<double> get_input_data (testify::GeneratedCase& gcase, std::string key)
{
	auto& inputs = gcase.inputs();
	auto it = inputs.find(key);
	if (inputs.end() == it)
	{
		throw std::runtime_error(key + " not found in input");
	}
	assert(it->second.has_ddoubles());
	const testify::Doubles& arr = it->second.ddoubles();
	auto temp = arr.data();
	return std::vector<double>(temp.begin(), temp.end());
}


static std::vector<double> get_output_data (testify::GeneratedCase& gcase, std::string key)
{
	auto& outputs = gcase.outputs();
	auto it = outputs.find(key);
	if (outputs.end() == it)
	{
		throw std::runtime_error(key + " not found in output");
	}
	assert(it->second.has_ddoubles());
	const testify::Doubles& arr = it->second.ddoubles();
	auto temp = arr.data();
	return std::vector<double>(temp.begin(), temp.end());
}


static void unary_op (antero::Testament* test, std::string tname,
	std::function<ade::TensptrT(ade::TensptrT&)> op)
{
	testify::GeneratedCase gcase = test->get("REGRESS::" + tname);
	ade::Shape shape = get_shape(gcase);
	std::vector<double> data = get_input_data(gcase, "data");
	std::vector<double> resdata = get_output_data(gcase, "unary_out");
	std::vector<double> gresdata = get_output_data(gcase, "unary_ga");

	ade::TensptrT leaf = llo::get_variable<double>(data, shape);
	ade::TensptrT res = op(leaf);
	ade::TensptrT gres = llo::derive(res, leaf.get());

	llo::GenericData resgd = llo::eval(res, age::DOUBLE);
	llo::GenericData gresgd = llo::eval(gres, age::DOUBLE);

	double* resptr = (double*) resgd.data_.get();
	double* gresptr = (double*) gresgd.data_.get();

	std::vector<double> resd(resptr, resptr + resgd.shape_.n_elems());
	std::vector<double> gresd(gresptr, gresptr + gresgd.shape_.n_elems());

	EXPECT_DATA_EQ("res", resdata, resd);
	EXPECT_DATA_EQ("gres", gresdata, gresd);
}


static void binary_op (antero::Testament* test, std::string tname,
	std::function<ade::TensptrT(ade::TensptrT&,ade::TensptrT&)> op)
{
	testify::GeneratedCase gcase = test->get("REGRESS::" + tname);
	ade::Shape shape = get_shape(gcase);
	std::vector<double> data = get_input_data(gcase, "data");
	std::vector<double> data2 = get_input_data(gcase, "data2");
	std::vector<double> resdata = get_output_data(gcase, "binary_out");
	std::vector<double> gresdata = get_output_data(gcase, "binary_ga");
	std::vector<double> gresdata2 = get_output_data(gcase, "binary_gb");

	ade::TensptrT leaf = llo::get_variable<double>(data, shape);
	ade::TensptrT leaf2 = llo::get_variable<double>(data2, shape);
	ade::TensptrT res = op(leaf, leaf2);
	ade::TensptrT gres = llo::derive(res, leaf.get());
	ade::TensptrT gres2 = llo::derive(res, leaf2.get());

	llo::GenericData resgd = llo::eval(res, age::DOUBLE);
	llo::GenericData gresgd = llo::eval(gres, age::DOUBLE);
	llo::GenericData gresgd2 = llo::eval(gres2, age::DOUBLE);

	double* resptr = (double*) resgd.data_.get();
	double* gresptr = (double*) gresgd.data_.get();
	double* gresptr2 = (double*) gresgd2.data_.get();

	std::vector<double> resd(resptr, resptr + resgd.shape_.n_elems());
	std::vector<double> gresd(gresptr, gresptr + gresgd.shape_.n_elems());
	std::vector<double> gresd2(gresptr2, gresptr2 + gresgd2.shape_.n_elems());

	EXPECT_DATA_EQ("res", resdata, resd);
	EXPECT_DATA_EQ("gres", gresdata, gresd);
	EXPECT_DATA_EQ("gres2", gresdata2, gresd2);
}


int main (int argc, char** argv)
{
	// todo: make this configurable
	dora::ClientConfig cfg;
	cfg.host = "localhost:10000";
	cfg.cert = dora::read_keycert("certs/server.crt");
	antero::INIT(cfg);

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();

	antero::SHUTDOWN();
	return ret;
}


struct REGRESS : public antero::Testament {};


TEST_F(REGRESS, Abs)
{
	unary_op(this, "Abs", [](ade::TensptrT& a)
	{
		return age::abs(a);
	});
}


TEST_F(REGRESS, Neg)
{
	unary_op(this, "Neg", [](ade::TensptrT& a)
	{
		return age::neg(a);
	});
}


TEST_F(REGRESS, Sin)
{
	unary_op(this, "Sin", [](ade::TensptrT& a)
	{
		return age::sin(a);
	});
}


TEST_F(REGRESS, Cos)
{
	unary_op(this, "Cos", [](ade::TensptrT& a)
	{
		return age::cos(a);
	});
}

TEST_F(REGRESS, Tan)
{
	unary_op(this, "Tan", [](ade::TensptrT& a)
	{
		return age::tan(a);
	});
}


TEST_F(REGRESS, Exp)
{
	unary_op(this, "Exp", [](ade::TensptrT& a)
	{
		return age::exp(a);
	});
}


TEST_F(REGRESS, Log)
{
	unary_op(this, "Log", [](ade::TensptrT& a)
	{
		return age::log(a);
	});
}


TEST_F(REGRESS, Sqrt)
{
	unary_op(this, "Sqrt", [](ade::TensptrT& a)
	{
		return age::sqrt(a);
	});
}


TEST_F(REGRESS, Pow)
{
	binary_op(this, "Pow", [](ade::TensptrT& a, ade::TensptrT& b)
	{
		return age::pow(a, b);
	});
}


TEST_F(REGRESS, Add)
{
	binary_op(this, "Add", [](ade::TensptrT& a, ade::TensptrT& b)
	{
		return age::add(a, b);
	});
}


TEST_F(REGRESS, Sub)
{
	binary_op(this, "Sub", [](ade::TensptrT& a, ade::TensptrT& b)
	{
		return age::sub(a, b);
	});
}


TEST_F(REGRESS, Mul)
{
	binary_op(this, "Mul", [](ade::TensptrT& a, ade::TensptrT& b)
	{
		return age::mul(a, b);
	});
}


TEST_F(REGRESS, Div)
{
	binary_op(this, "Div", [](ade::TensptrT& a, ade::TensptrT& b)
	{
		return age::div(a, b);
	});
}


TEST_F(REGRESS, Matmul)
{
	testify::GeneratedCase gcase = get("REGRESS::Matmul");
	ade::Shape ashape;
	ade::Shape bshape;
	{
		auto& inputs = gcase.inputs();
		auto it = inputs.find("ashape");
		if (inputs.end() == it)
		{
			throw std::runtime_error("ashape not found");
		}
		assert(it->second.has_dint64s());
		const testify::Int64s& arr = it->second.dint64s();
		auto temp = arr.data();
		std::vector<ade::DimT> slist(temp.begin(), temp.end());
		ashape = ade::Shape(slist);

		auto bit = inputs.find("bdim");
		if (inputs.end() == bit)
		{
			throw std::runtime_error("bdim not found");
		}
		assert(bit->second.has_dint64s());
		const testify::Int64s& barr = bit->second.dint64s();
		ade::DimT bdim = barr.data()[0];
		bshape = ade::Shape({bdim, slist[0]});
	}

	std::vector<double> data = get_input_data(gcase, "data");
	std::vector<double> data2 = get_input_data(gcase, "data2");
	std::vector<double> resdata = get_output_data(gcase, "matmul_out");
	std::vector<double> gresdata = get_output_data(gcase, "matmul_ga");
	std::vector<double> gresdata2 = get_output_data(gcase, "matmul_gb");

	ade::TensptrT leaf = llo::get_variable<double>(data, ashape);
	ade::TensptrT leaf2 = llo::get_variable<double>(data2, bshape);
	ade::TensptrT res = age::matmul(leaf, leaf2);
	ade::TensptrT gres = llo::derive(res, leaf.get());
	ade::TensptrT gres2 = llo::derive(res, leaf2.get());

	llo::GenericData resgd = llo::eval(res, age::DOUBLE);
	llo::GenericData gresgd = llo::eval(gres, age::DOUBLE);
	llo::GenericData gresgd2 = llo::eval(gres2, age::DOUBLE);

	double* resptr = (double*) resgd.data_.get();
	double* gresptr = (double*) gresgd.data_.get();
	double* gresptr2 = (double*) gresgd2.data_.get();

	std::vector<double> resd(resptr, resptr + resgd.shape_.n_elems());
	size_t totalga = gresgd.shape_.n_elems();
	std::vector<double> gresd(gresptr, gresptr + totalga);
	size_t totalgb = gresgd2.shape_.n_elems();
	std::vector<double> gresd2(gresptr2, gresptr2 + totalgb);

	EXPECT_DATA_APPROX("res", resdata, resd);
	EXPECT_DATA_APPROX("gres", gresdata, gresd);
	EXPECT_DATA_APPROX("gres2", gresdata2, gresd2);
}
