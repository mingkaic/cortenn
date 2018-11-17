
#ifndef DISABLE_DATA_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/data.hpp"
#include "llo/eval.hpp"


struct DATA : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(DATA, MismatchSize)
{
	simple::SessionT sess = get_session("DATA::MismatchSize");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	std::vector<double> data = sess->get_double("data", shape.n_elems() - 1);

	std::stringstream ss;
	ss << "cannot create variable with data size " << data.size() <<
		" against shape " << shape.to_string();
	EXPECT_FATAL(llo::get_variable<double>(data, shape), ss.str().c_str());
}


TEST_F(DATA, SourceRetype)
{
	simple::SessionT sess = get_session("DATA::SourceRetype");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);

	size_t n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n);
	ade::Tensorptr ptr = llo::get_variable<double>(data, shape);

	llo::GenericData gd = llo::eval(ptr, age::UINT16);
	ASSERT_EQ(age::UINT16, gd.dtype_);
	std::vector<ade::DimT> gotslist(gd.shape_.begin(), gd.shape_.end());
	EXPECT_ARREQ(slist, gotslist);

	uint16_t* gotdata = (uint16_t*) gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ((uint16_t) data[i], gotdata[i]);
	}
}


TEST_F(DATA, PlaceHolder)
{
	simple::SessionT sess = get_session("DATA::Placeholder");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	size_t n = shape.n_elems();
	llo::VarptrT pl(llo::get_variable<double>(shape));

	llo::GenericData uninit_gd = llo::eval(ade::Tensorptr(pl), age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, uninit_gd.dtype_);
	std::vector<ade::DimT> uninit_slist(uninit_gd.shape_.begin(), uninit_gd.shape_.end());
	EXPECT_ARREQ(slist, uninit_slist);

	double* uninit_data = (double*) uninit_gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(0, uninit_data[i]);
	}

	std::vector<double> data = sess->get_double("data", n);
	*pl = data;
	llo::GenericData gd = llo::eval(ade::Tensorptr(pl), age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gd.dtype_);
	std::vector<ade::DimT> gotslist(gd.shape_.begin(), gd.shape_.end());
	EXPECT_ARREQ(slist, gotslist);

	double* gotdata = (double*) gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(data[i], gotdata[i]);
	}
}


#endif // DISABLE_DATA_TEST
