
#ifndef DISABLE_VARIABLE_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "exam/exam.hpp"

#include "llo/variable.hpp"
#include "llo/eval.hpp"


TEST(DATA, MismatchSize)
{
	std::vector<ade::DimT> slist = {4, 2, 3};
	ade::Shape shape(slist);
	std::vector<double> data = {
		41, 29, 86, 43, 12, 55, 68, 87, 16, 92, 26, 28,
		13, 1, 62, 9, 27, 10, 23, 70, 80, 67, 96, 22, 24
	};

	std::stringstream ss;
	ss << "cannot create variable with data size " << data.size() <<
		" against shape " << shape.to_string();
	EXPECT_FATAL(llo::Variable<double>::get(data, shape), ss.str().c_str());
}


TEST(DATA, SourceRetype)
{
	std::vector<ade::DimT> slist = {3, 3, 3};
	ade::Shape shape(slist);

	size_t n = shape.n_elems();
	std::vector<double> data = {
		16, 51, 12, 55, 69, 10, 52, 86, 95,
		6, 78, 18, 100, 11, 52, 66, 55, 30,
		80, 81, 36, 26, 63, 78, 80, 31, 37
	};
	ade::TensptrT ptr(llo::Variable<double>::get(data, shape));

	llo::TensptrT<uint16_t> gd = llo::eval<uint16_t>(ptr);
	auto gotslist = gd->dimensions();
	EXPECT_ARREQ(slist, gotslist);

	uint16_t* gotdata = (uint16_t*) gd->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ((uint16_t) data[i], gotdata[i]);
	}
}


TEST(DATA, PlaceHolder)
{
	std::vector<ade::DimT> slist = {2, 5, 2};
	ade::Shape shape(slist);
	size_t n = shape.n_elems();
	llo::VarptrT<double> pl(llo::Variable<double>::get(shape));

	llo::TensptrT<double> uninit_gd = llo::eval<double>(ade::TensptrT(pl));
	auto uninit_slist = uninit_gd->dimensions();
	EXPECT_ARREQ(slist, uninit_slist);

	double* uninit_data = (double*) uninit_gd->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(0, uninit_data[i]);
	}

	std::vector<double> data = {
		33, 80, 95, 40, 77, 70, 42, 31, 58, 53,
		48, 77, 58, 64, 83, 64, 6, 24, 16, 9
	};
	*pl = data;
	llo::TensptrT<double> gd = llo::eval<double>(ade::TensptrT(pl));
	auto gotslist = gd->dimensions();
	EXPECT_ARREQ(slist, gotslist);

	double* gotdata = (double*) gd->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(data[i], gotdata[i]);
	}
}


#endif // DISABLE_VARIABLE_TEST
