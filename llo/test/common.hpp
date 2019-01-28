#include "fmts/fmts.hpp"

#define ASSERT_ARREQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	ASSERT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_ARREQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	EXPECT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_FATAL(EVENT, MSG) try { EVENT; FAIL() << \
	"did not expect " << #EVENT << " to succeed"; } \
	catch (std::runtime_error& e) { EXPECT_STREQ(MSG, e.what()); }
