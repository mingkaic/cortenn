
#ifndef DISABLE_OPTIMIZATION_TEST

#include <list>

#include "gtest/gtest.h"

#include "dbg/stream/ade_csv.hpp"

#include "testutil/common.hpp"

#include "exam/exam.hpp"

#include "llo/opt/zero_prune.hpp"
#include "llo/opt/one_prune.hpp"
#include "llo/opt/const_merge.hpp"
#include "llo/opt/ops_merge.hpp"
#include "llo/opt/ops_reuse.hpp"

#include "llo/generated/api.hpp"

#include "llo/constant.hpp"


TEST(OPTIMIZATION, zero_prune_singles)
{
	ade::TensptrT zero(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));

	auto got0 = llo::zero_prune({age::sin(zero)})[0];
	EXPECT_STREQ("0", got0->to_string().c_str());

	auto got1 = llo::zero_prune({age::cos(zero)})[0];
	EXPECT_STREQ("1", got1->to_string().c_str());

	auto got3 = llo::zero_prune({age::sum({one, zero, two})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_STREQ("", compare_graph(ss, got3).c_str());
	}

	auto gotn1 = llo::zero_prune({age::sub(zero, one)})[0];
	{
		std::stringstream ss;
		ss <<
			"(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_STREQ("", compare_graph(ss, gotn1).c_str());
	}

	auto got2 = llo::zero_prune({age::sub(two, zero)})[0];
	EXPECT_STREQ("2", got2->to_string().c_str());

	auto got00 = llo::zero_prune({age::prod({two, zero, one})})[0];
	EXPECT_STREQ("0", got00->to_string().c_str());

	auto got000 = llo::zero_prune({age::div(zero, two)})[0];
	EXPECT_STREQ("0", got000->to_string().c_str());

	EXPECT_FATAL(llo::zero_prune({age::div(one, zero)}), "cannot DIV by zero");

	auto gotnormal = llo::zero_prune({age::max({two, zero})})[0];
	{
		std::stringstream ss;
		ss <<
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0[1\\1\\1\\1\\1\\1\\1\\1])";
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
		" |   |   |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |       `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |           `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |           `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(2[1\\1\\1\\1\\1\\1\\1\\1])";
	auto compare_str = compare_graph(ss, opt_nocascades);
	EXPECT_EQ(0, compare_str.size()) << compare_str;

	auto got0 = age::tan(zero);
	auto opt_cascades = llo::zero_prune({age::pow(nocascades, got0)})[0];
	EXPECT_STREQ("1", opt_cascades->to_string().c_str());
}


TEST(OPTIMIZATION, one_prune_singles)
{
	ade::TensptrT zero(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));

	auto got0 = llo::one_prune({age::log(one)})[0];
	EXPECT_STREQ("0", got0->to_string().c_str());

	auto got1 = llo::one_prune({age::sqrt(one)})[0];
	EXPECT_STREQ("1", got1->to_string().c_str());

	auto got02 = llo::one_prune({age::prod({one, zero, two})})[0];
	{
		std::stringstream ss;
		ss <<
			"(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_STREQ("", compare_graph(ss, got02).c_str());
	}

	auto got2 = llo::one_prune({age::div(two, one)})[0];
	EXPECT_STREQ("2", got2->to_string().c_str());

	auto gottoo = llo::one_prune({age::pow(two, one)})[0];
	EXPECT_STREQ("2", gottoo->to_string().c_str());

	auto gotone = llo::one_prune({age::pow(one, two)})[0];
	EXPECT_STREQ("1", gotone->to_string().c_str());

	auto gotnormal = llo::one_prune({age::max({two, one})})[0];
	{
		std::stringstream ss;
		ss <<
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_STREQ("", compare_graph(ss, gotnormal).c_str());
	}
}


TEST(OPTIMIZATION, one_prune_graph)
{
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));

	auto got0 = age::log(one);
	auto got1 = age::sqrt(one);
	auto got3 = age::prod({one, got0, two});
	auto got00 = age::pow(one, two);
	auto got = age::max({two, one});

	auto too = age::add(got1, age::prod({got1, got00}));
	auto got11 = age::pow(two, one);

	auto m = age::min({got1, too, got11});
	auto root = age::sub(age::pow(m, age::div(got3, got)), two);

	auto opt = llo::one_prune({root})[0];
	std::stringstream ss;
	ss <<
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(2[1\\1\\1\\1\\1\\1\\1\\1])";
	auto compare_str = compare_graph(ss, opt);
	EXPECT_EQ(0, compare_str.size()) << compare_str;
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
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, got1123);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// don't merge different nnary
	auto got1_12_3 = llo::ops_merge({age::sum({one, age::max({one, two}), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, got1_12_3);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// merge single unary argument of nnary
	auto got213 = llo::ops_merge({age::sum({two, age::max({one}), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, got213);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// don't merge single unary argument of non-nnary
	auto got2_1_3 = llo::ops_merge({age::sum({two, age::tan(one), three})})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(TAN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
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
			" `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0[3\\4\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
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
			" `--(1[3\\4\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0[3\\4\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, got10);
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	// merge redundent double reduced argument
	auto got0 = llo::ops_merge({age::reduce_sum(age::reduce_sum(zero))})[0];
	{
		std::stringstream ss;
		ss <<
			"(SUM[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0[3\\4\\1\\1\\1\\1\\1\\1])\n";
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
			"     `--(0[3\\4\\1\\1\\1\\1\\1\\1])\n";
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
			" |   `--(0[3\\4\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n";
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
		" |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(PROD[4\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(0[3\\4\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(PROD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
	auto compare_str = compare_graph(ss, root);
	EXPECT_EQ(0, compare_str.size()) << compare_str;
}


TEST(OPTIMIZATION, const_merge_graph)
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
	auto root = age::sub(age::pow(m, age::div(got3, gotn1)), got2);

	auto opt = llo::const_merge({root})[0];
	EXPECT_STREQ("-1", opt->to_string().c_str());
}


TEST(OPTIMIZATION, reuse_op_graph)
{
	ade::TensptrT zero(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT zero2(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT zero3(llo::Constant::get(0, ade::Shape()));
	ade::TensptrT one(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT one2(llo::Constant::get(1, ade::Shape()));
	ade::TensptrT two(llo::Constant::get(2, ade::Shape()));
	ade::TensptrT two2(llo::Constant::get(2, ade::Shape()));

	ade::TensptrT root;
	{
		auto got1 = age::cos(zero);
		auto got3 = age::sum({one, zero, two2});
		auto gotn1 = age::sub(zero2, one2);
		auto got2 = age::sub(two, zero3);
		auto got22 = age::max({two, zero2});

		auto too = age::add(zero, age::prod({got1, got22}));
		auto got11 = age::pow(got2, zero3);

		auto m = age::min({got22, got1, too, got11});
		root = age::sub(age::pow(m, age::div(got3, gotn1)), got2);
	}

	ade::TensptrT subroot;
	{
		auto other_got1 = age::cos(zero);
		auto got22 = age::max({two2, zero3});
		subroot = age::prod({other_got1, got22});
	}

	ade::TensptrT copyroot;
	{
		auto got1 = age::cos(zero);
		auto got3 = age::sum({one, zero, two2});
		auto gotn1 = age::sub(zero2, one2);
		auto got2 = age::sub(two, zero3);
		auto got22 = age::max({two, zero2});

		auto too = age::add(zero, age::prod({got1, got22}));
		auto got11 = age::pow(got2, zero3);

		auto m = age::min({got22, got1, too, got11});
		copyroot = age::sub(age::pow(m, age::div(got3, gotn1)), got2);
	}

	ade::TensptrT splitroot;
	{
		auto got1 = age::cos(zero);
		auto got3 = age::sum({one, zero, two2});
		auto gotn1 = age::sub(zero2, one2);
		auto got2 = age::sub(two, zero3);
		auto got22 = age::max({two, zero2});

		auto too = age::div(got2, age::prod({got1, got22}));
		auto got11 = age::eq(too, gotn1);

		splitroot = age::prod({got11, got1, too, got3});
	}

	auto opts = llo::ops_reuse({subroot, root, splitroot, copyroot});
	auto opt_subroot = opts[0];
	auto opt_root = opts[1];
	auto opt_splitroot = opts[2];
	auto opt_copyroot = opts[3];

	ASSERT_NE(nullptr, opt_subroot);
	ASSERT_NE(nullptr, opt_root);
	ASSERT_NE(nullptr, opt_splitroot);
	ASSERT_NE(nullptr, opt_copyroot);

	std::stringstream ss;
	CSVEquation ceq;
	opt_subroot->accept(ceq);
	opt_root->accept(ceq);
	opt_splitroot->accept(ceq);
	opt_copyroot->accept(ceq);
	ceq.to_stream(ss);

	std::list<std::string> expectlines =
	{
		"0:PROD,1:COS,0,white",
		"1:COS,2:0,0,white",
		"0:PROD,3:MAX,1,white",
		"3:MAX,4:2,0,white",
		"3:MAX,2:0,1,white",
		"5:SUB,6:POW,0,white",
		"6:POW,7:MIN,0,white",
		"7:MIN,3:MAX,0,white",
		"7:MIN,1:COS,1,white",
		"7:MIN,8:SUM,2,white",
		"8:SUM,2:0,0,white",
		"8:SUM,0:PROD,1,white",
		"7:MIN,9:POW,3,white",
		"9:POW,10:SUB,0,white",
		"10:SUB,4:2,0,white",
		"10:SUB,2:0,1,white",
		"9:POW,2:0,1,white",
		"6:POW,11:DIV,1,white",
		"11:DIV,12:SUM,0,white",
		"12:SUM,13:1,0,white",
		"12:SUM,2:0,1,white",
		"12:SUM,4:2,2,white",
		"11:DIV,14:SUB,1,white",
		"14:SUB,2:0,0,white",
		"14:SUB,13:1,1,white",
		"5:SUB,10:SUB,1,white",
		"15:PROD,16:EQ,0,white",
		"16:EQ,17:DIV,0,white",
		"17:DIV,10:SUB,0,white",
		"17:DIV,0:PROD,1,white",
		"16:EQ,14:SUB,1,white",
		"15:PROD,1:COS,1,white",
	};
	expectlines.sort();
	std::list<std::string> gotlines;
	std::string line;
	while (std::getline(ss, line))
	{
		gotlines.push_back(line);
	}
	gotlines.sort();
	std::vector<std::string> diffs;
	std::set_difference(expectlines.begin(), expectlines.end(),
		gotlines.begin(), gotlines.end(), std::back_inserter(diffs));

	std::stringstream diffstr;
	for (auto diff : diffs)
	{
		diffstr << diff << "\n";
	}
	std::string diffmsg = diffstr.str();
	EXPECT_EQ(0, diffmsg.size()) << "mismatching edges:\n" << diffmsg;
}


#endif // DISABLE_OPTIMIZATION_TEST
