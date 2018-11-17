#include "err/log.hpp"
#include "err/string.hpp"
#include "ade/shape.hpp"
#include "ade/functor.hpp"

#include "simple/jack.hpp"

struct TestLogger : public err::iLogger
{
	static std::string latest_warning_;
	static std::string latest_error_;

	void warn (std::string msg) const override
	{
		latest_warning_ = msg;
	}

	void error (std::string msg) const override
	{
		latest_error_ = msg;
	}

	void fatal (std::string msg) const override
	{
		throw std::runtime_error(msg);
	}
};

extern std::shared_ptr<TestLogger> tlogger;

const size_t nelem_limit = 32456;

#define ASSERT_ARREQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	err::to_stream(arrs, ARR.begin(), ARR.end());\
	err::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	ASSERT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_ARREQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	err::to_stream(arrs, ARR.begin(), ARR.end());\
	err::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	EXPECT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_FATAL(EVENT, MSG) try { EVENT; } catch (std::runtime_error& e) {\
	EXPECT_STREQ(MSG, e.what()); }

std::vector<ade::DimT> get_shape_n (simple::SessionT& sess, size_t n, std::string label);

std::vector<ade::DimT> get_shape (simple::SessionT& sess, std::string label);

std::vector<ade::DimT> get_zeroshape (simple::SessionT& sess, std::string label);

std::vector<ade::DimT> get_longshape (simple::SessionT& sess, std::string label);

std::vector<ade::DimT> get_incompatible (simple::SessionT& sess,
	std::vector<ade::DimT> inshape, std::string label);

void int_verify (simple::SessionT& sess, std::string key,
	std::vector<int32_t> data, std::function<void()> verify);

void double_verify (simple::SessionT& sess, std::string key,
	std::vector<double> data, std::function<void()> verify);
