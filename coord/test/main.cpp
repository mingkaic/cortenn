#include "gtest/gtest.h"

#include "coord/coord.hpp"


int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


#ifndef DISABLE_COORD_TEST


TEST(COORD, EigenMap)
{
	coord::EigenMap identity([](Eigen::MatrixXd& m)
	{
		for (uint8_t i = 0; i < ade::mat_dim; ++i)
		{
			m(i, i) = 1;
		}
	});

	EXPECT_TRUE(identity.is_bijective());

	coord::EigenMap reduce([](Eigen::MatrixXd& m)
	{
		for (uint8_t i = 0; i < ade::mat_dim; ++i)
		{
			m(i, i) = 1;
		}
		m(3, 3) = 0.5;
	});

	EXPECT_FALSE(reduce.is_bijective());
}


#endif // DISABLE_COORD_TEST
