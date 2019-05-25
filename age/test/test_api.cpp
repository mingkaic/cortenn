
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "age/test/grader_dep.hpp"
#include "age/generated/api.hpp"


TEST(AGE, Api)
{
	ade::TensptrT carrot(age::goku(16));
	MockTensor* kakarot = dynamic_cast<MockTensor*>(carrot.get());
	EXPECT_NE(nullptr, kakarot);
	ade::Shape shape = kakarot->shape();
	EXPECT_EQ(16, kakarot->scalar_);
	EXPECT_EQ(16, shape.n_elems());
	EXPECT_EQ(16, shape.at(0));

	ade::TensptrT vegetable(
		age::vegeta(ade::TensptrT(
			new MockTensor(1, ade::Shape({1, 1, 31}))), 2));
	MockTensor* planet = dynamic_cast<MockTensor*>(vegetable.get());
	EXPECT_NE(nullptr, planet);
	ade::Shape vshape = planet->shape();
	EXPECT_EQ(2, planet->scalar_);
	EXPECT_EQ(31, vshape.n_elems());
	EXPECT_EQ(31, vshape.at(0));

	ade::TensptrT vegetable2(
		age::vegeta(2, {ade::TensptrT(
			new MockTensor(1, ade::Shape({1, 1, 31})))}));
	MockTensor* planet2 = dynamic_cast<MockTensor*>(vegetable2.get());
	EXPECT_NE(nullptr, planet2);
	ade::Shape vshape2 = planet2->shape();
	EXPECT_EQ(2, planet2->scalar_);
	EXPECT_EQ(31, vshape2.n_elems());
	EXPECT_EQ(31, vshape2.at(0));
}


#endif // DISABLE_API_TEST
