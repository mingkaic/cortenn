
#ifndef DISABLE_CAPI_TEST


#include "gtest/gtest.h"

#include "age/test/grader_dep.hpp"
#include "age/generated/capi.hpp"


TEST(AGE, CApi)
{
	// everything should be exactly the same as Api
	// except inputs and output types are different
	int64_t carrot = age_goku(16);
	MockTensor* kakarot = dynamic_cast<MockTensor*>(
		get_tens(carrot).get());
	EXPECT_NE(nullptr, kakarot);
	ade::Shape shape = kakarot->shape();
	EXPECT_EQ(16, kakarot->scalar_);
	EXPECT_EQ(16, shape.n_elems());
	EXPECT_EQ(16, shape.at(0));

	int64_t var = register_tens(new MockTensor(1, ade::Shape({1, 1, 31})));
	int64_t vegetable = age_vegeta_1(var, 2);
	MockTensor* planet = dynamic_cast<MockTensor*>(
		get_tens(vegetable).get());
	EXPECT_NE(nullptr, planet);
	ade::Shape vshape = planet->shape();
	EXPECT_EQ(2, planet->scalar_);
	EXPECT_EQ(31, vshape.n_elems());
	EXPECT_EQ(31, vshape.at(0));

	int64_t varr[1] = {var};
	int64_t vegetable2 = age_vegeta(2, varr, 1);
	MockTensor* planet2 = dynamic_cast<MockTensor*>(
		get_tens(vegetable2).get());
	EXPECT_NE(nullptr, planet2);
	ade::Shape vshape2 = planet2->shape();
	EXPECT_EQ(2, planet2->scalar_);
	EXPECT_EQ(31, vshape2.n_elems());
	EXPECT_EQ(31, vshape2.at(0));
}


#endif // DISABLE_CAPI_TEST
