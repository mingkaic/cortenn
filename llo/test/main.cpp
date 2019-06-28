
#include "gtest/gtest.h"

#include "exam/exam.hpp"

int main (int argc, char** argv)
{
	set_logger(std::static_pointer_cast<logs::iLogger>(exam::tlogger));

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
