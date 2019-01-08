
#ifndef DISABLE_OPTIMIZATION_TEST


#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "llo/test/common.hpp"

#include "llo/generated/api.hpp"
#include "llo/zprune.hpp"


static inline void ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int,int>(std::isspace))));
}


static inline void rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int,int>(std::isspace))).base(), s.end());
}


static inline void trim(std::string &s)
{
	ltrim(s);
	rtrim(s);
}


#define TREE_EQ(expectstr, root)\
{\
	PrettyEquation artist;\
	artist.showshape_ = true;\
	std::stringstream gotstr;\
	artist.print(gotstr, root);\
	std::string expect;\
	std::string got;\
	std::string line;\
	while (std::getline(expectstr, line))\
	{\
		trim(line);\
		if (line.size() > 0)\
		{\
			expect += line + "\n";\
		}\
	}\
	while (std::getline(gotstr, line))\
	{\
		trim(line);\
		if (line.size() > 0)\
		{\
			got += line + "\n";\
		}\
	}\
	EXPECT_STREQ(expect.c_str(), got.c_str());\
}


TEST(OPTIMIZATION, zprune_singles)
{
    auto zero = llo::get_scalar(0, ade::Shape());
    auto one = llo::get_scalar(1, ade::Shape());
    auto two = llo::get_scalar(2, ade::Shape());

    auto got0 = llo::zero_prune(age::sin(zero));
    EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got0->to_string().c_str());

    auto got1 = llo::zero_prune(age::cos(zero));
    EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", got1->to_string().c_str());

    auto got3 = llo::zero_prune(age::sum({one, zero, two}));
    {
        std::stringstream ss;
        ss <<
            "(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
            " `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
            " `--(2([1\\1\\1\\1\\1\\1\\1\\1]))";
        TREE_EQ(ss, got3);
    }

    auto gotn1 = llo::zero_prune(age::sub(zero, one));
    {
        std::stringstream ss;
        ss <<
            "(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
            " `--(1([1\\1\\1\\1\\1\\1\\1\\1]))";
        TREE_EQ(ss, gotn1);
    }

    auto got2 = llo::zero_prune(age::sub(two, zero));
    EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", got2->to_string().c_str());

    auto got00 = llo::zero_prune(age::prod({two, zero, one}));
    EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got00->to_string().c_str());

    auto got000 = llo::zero_prune(age::div(zero, two));
    EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got000->to_string().c_str());

    EXPECT_FATAL(llo::zero_prune(age::div(one, zero)), "cannot DIV by zero");

    auto gotnormal = llo::zero_prune(age::max({two, zero}));
    {
        std::stringstream ss;
        ss <<
            "(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
            " `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
            " `--(0([1\\1\\1\\1\\1\\1\\1\\1]))";
        TREE_EQ(ss, gotnormal);
    }
}


TEST(OPTIMIZATION, zprune_graph)
{
    auto zero = llo::get_scalar(0, ade::Shape());
    auto one = llo::get_scalar(1, ade::Shape());
    auto two = llo::get_scalar(2, ade::Shape());

    auto got1 = age::cos(zero);
    auto got3 = age::sum({one, zero, two});
    auto gotn1 = age::sub(zero, one);
    auto got2 = age::sub(two, zero);
    auto got22 = age::max({two, zero});

    auto too = age::add(zero, age::prod({got1, got22}));
    auto got11 = age::pow(got2, zero);

    auto m = age::min({got22, got1, got11});
    auto nocascades = age::sub(age::pow(m, age::div(got3, gotn1)), got2);

    auto opt_nocascades = llo::zero_prune(nocascades);
    std::stringstream ss;
    ss <<
        "(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " |   |   |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " |   |   |   `--(0([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " |   |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " |   |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " |       `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " |       |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " |       |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " |       `--(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
        " |           `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
        " `--(2([1\\1\\1\\1\\1\\1\\1\\1]))";
    TREE_EQ(ss, opt_nocascades);

    auto got0 = age::tan(zero);
    auto opt_cascades = llo::zero_prune(age::pow(nocascades, got0));
    EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", opt_cascades->to_string().c_str());
}


#endif // DISABLE_OPTIMIZATION_TEST
