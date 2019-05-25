
#ifndef DISABLE_GRADER_TEST


#include "gtest/gtest.h"

#include "age/test/grader_dep.hpp"
#include "age/generated/grader.hpp"


TEST(AGE, GraderEminem)
{
	auto mock = new MockTensor(1, ade::Shape());
	ade::TensptrT arg(mock);
	ade::Functor* fwd = ade::Functor::get(ade::Opcode{"EMINEM", age::EMINEM},
		{ade::identity_map(arg)});
	size_t idx = 42;
	// bwd is never used so use whatever
	age::chain_rule(fwd, ade::identity_map(arg), {arg}, idx);
	EXPECT_EQ(idx, mock->scalar_);
	delete fwd;
}


TEST(AGE, GraderKhaled)
{
	auto mock = new MockTensor(1, ade::Shape());
	ade::TensptrT arg(mock);
	ade::Functor* fwd = ade::Functor::get(ade::Opcode{"KHALED", age::KHALED},
		{ade::identity_map(arg)});
	size_t idx = 63;
	// bwd is never used so use whatever
	age::chain_rule(fwd, ade::identity_map(arg), {arg}, idx);
	EXPECT_EQ(idx + khaled_constant, mock->scalar_);
	delete fwd;
}


#endif // DISABLE_GRADER_TEST
