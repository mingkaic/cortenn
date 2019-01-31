
#ifndef DISABLE_OPTIMIZATION_TEST


#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "testutil/common.hpp"

#include "llo/opt/zero_prune.hpp"
#include "llo/opt/ops_merge.hpp"

#include "llo/generated/api.hpp"

#include "llo/constant.hpp"


TEST(OPTIMIZATION, zero_prune_singles)
{
	ade::TensptrT zero(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));

	auto got0 = llo::zero_prune({age::sin(zero)})[0];
	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got0->to_string().c_str());

	auto got1 = llo::zero_prune({age::cos(zero)})[0];
	EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", got1->to_string().c_str());

	auto got3 = llo::zero_prune({age::sum({one, zero, two})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))";
		EXPECT_STREQ("", compare_graph(ss, got3).c_str());
	}

	auto gotn1 = llo::zero_prune({age::sub(zero, one)})[0];
	{
		std::stringstream ss;
		ss <<
			"(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))";
		EXPECT_STREQ("", compare_graph(ss, gotn1).c_str());
	}

	auto got2 = llo::zero_prune({age::sub(two, zero)})[0];
	EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", got2->to_string().c_str());

	auto got00 = llo::zero_prune({age::prod({two, zero, one})})[0];
	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got00->to_string().c_str());

	auto got000 = llo::zero_prune({age::div(zero, two)})[0];
	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got000->to_string().c_str());

	EXPECT_FATAL(llo::zero_prune({age::div(one, zero)}), "cannot DIV by zero");

	auto gotnormal = llo::zero_prune({age::max({two, zero})})[0];
	{
		std::stringstream ss;
		ss <<
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(0([1\\1\\1\\1\\1\\1\\1\\1]))";
		EXPECT_STREQ("", compare_graph(ss, gotnormal).c_str());
	}
}


TEST(OPTIMIZATION, zero_prune_graph)
{
	ade::TensptrT zero(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));

	auto got1 = age::cos(zero);
	auto got3 = age::sum({one, zero, two});
	auto gotn1 = age::sub(zero, one);
	auto got2 = age::sub(two, zero);
	auto got22 = age::max({two, zero});

	auto too = age::add(zero, age::prod({got1, got22}));
	auto got11 = age::pow(got2, zero);

	auto m = age::min({got22, got1, too, got11});
	auto nocascades = age::sub(age::pow(m, age::div(got3, gotn1)), got2);

	auto opt_nocascades = llo::zero_prune({nocascades})[0];
	std::stringstream ss;
	ss <<
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   |   `--(0([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |       `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |           `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   |           `--(0([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |       |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |       `--(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))";
	auto compare_str = compare_graph(ss, opt_nocascades);
	EXPECT_EQ(0, compare_str.size()) << compare_str;

	auto got0 = age::tan(zero);
	auto opt_cascades = llo::zero_prune({age::pow(nocascades, got0)})[0];
	EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", opt_cascades->to_string().c_str());
}


TEST(OPTIMIZATION, ops_merge_singles)
{
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));
	ade::TensptrT three(llo::Constant::get(3, ade::Shape()));

	// merge same consecutive nnary
	auto got1123 = llo::ops_merge({age::sum({one, age::add(one, two), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, got1123);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// don't merge different nnary
	auto got1_12_3 = llo::ops_merge({age::sum({one, age::max({one, two}), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, got1_12_3);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// merge single unary argument of nnary
	auto got213 = llo::ops_merge({age::sum({two, age::max({one}), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, got213);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// don't merge single unary argument of non-nnary
	auto got2_1_3 = llo::ops_merge({age::sum({two, age::tan(one), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(TAN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, got2_1_3);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	ade::TensptrT zero(llo::Variable<double>::get(ade::Shape({3, 4}), "0"));
	// merge reduced argument
	auto got2103 = llo::ops_merge({age::sum({two, one, age::reduce_sum(zero), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, got2103);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// merge reduced sum
	ade::TensptrT shaped_one(llo::Constant::get(1, ade::Shape({3, 4})));
	auto got10 = llo::ops_merge({age::reduce_sum(age::sum({shaped_one, zero}))})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(0([3\\4\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, got10);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// merge redundent double reduced argument
	auto got0 = llo::ops_merge({age::reduce_sum(age::reduce_sum(zero))})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n";
		auto compare_str = compare_graph(ss, got0);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// don't merge non-redundent double reduced argument
	auto got_0 = llo::ops_merge({age::reduce_sum(age::reduce_sum(zero, 1), 0)})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(SUM[3\\1\\1\\1\\1\\1\\1\\1])\n" <<
			"     `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n";
		auto compare_str = compare_graph(ss, got_0);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// don't merge prod-reduced_sum
	auto got_0_1 = llo::ops_merge({age::prod({age::reduce_sum(zero), one})})[0];
	{
		std::stringstream ss;
		ss <<
			"(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n";
		auto compare_str = compare_graph(ss, got_0_1);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}
}


TEST(OPTIMIZATION, ops_merge_graph)
{
	ade::TensptrT zero(llo::Variable<double>::get(ade::Shape({3, 4}), "0"));
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));
	ade::TensptrT three(llo::Constant::get(3, ade::Shape()));

	auto got1 = age::cos(three);
	auto got3 = age::prod({one, three, two});
	auto gotn1 = age::sub(three, one);
	auto got2 = age::sub(two, three);
	auto got22 = age::min({two, three});

	auto too = age::mul(age::reduce_prod(age::reduce_prod_1d(zero, 0), 0),
		age::reduce_prod(age::prod({got1, got22})));
	auto got11 = age::pow(got2, three);

	auto m = age::min({got22, got1, too, got11});
	auto root = llo::ops_merge({age::sub(
		age::min({m, age::div(got3, gotn1)}), got2)})[0];

	std::stringstream ss;
	ss <<
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(PROD[4\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |       `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |       |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |       |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |       `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" |           `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		" `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
		"     `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
	auto compare_str = compare_graph(ss, root);
	EXPECT_EQ(0, compare_str.size()) << compare_str;
}


#endif // DISABLE_OPTIMIZATION_TEST
