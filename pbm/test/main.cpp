#include "gtest/gtest.h"

#include "pbm/graph.pb.h"

int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
