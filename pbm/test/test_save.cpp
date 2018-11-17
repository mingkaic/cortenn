
#ifndef DISABLE_SAVE_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "llo/api.hpp"

#include "pbm/graph.hpp"


const std::string testdir = "pbm/data";


TEST(SAVE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/graph.pb";
	std::string got_pbfile = "got_graph.pb";
	tenncor::Graph graph;
	std::vector<llo::Variable> roots;

	// subtree one
	ade::Shape shape({3, 7});
	std::vector<uint8_t> odata = {69, 49, 43, 96, 38, 21, 6, 26, 26, 57, 46, 69,
		98, 66, 98, 84, 5, 78, 82, 95, 98};
	auto osrc = llo::Source<uint8_t>::get(shape, odata);

	std::vector<int16_t> odata2 = {18, 73, -60, -11, 47, -96};
	ade::Shape shape2({2, 3});
	auto osrc2 = llo::Source<int16_t>::get(shape2, odata2);

	{
		std::vector<uint64_t> data = {16, 65, 57, 11, 10, 17, 76, 47, 47, 44, 47, 14,
			9, 54, 35, 94, 15, 93, 43, 56, 50};
		auto src = llo::Source<uint64_t>::get(shape, data);

		ade::Shape shape3({2, 7});
		std::vector<uint32_t> odata3 = {6, 15, 30, 37, 4, 14, 89, 73, 84, 18, 49, 58, 26, 33};
		auto src2 = llo::Source<uint32_t>::get(shape3, odata3);

		auto dest = llo::sub(src2, llo::matmul(
			llo::div(llo::neg(osrc), llo::add(llo::sin(src), src)), osrc2));
		roots.push_back(dest);
	}

	// subtree two
	{
		ade::Shape mshape({11, 3});
		std::vector<float> data = {90, 34, 15, 21, 69, 24, 34, 16, 18, 51, 59,
			80, 34, 60, 82, 42, 4, 68, 99, 90, 98, 1, 98, 81, 43, 48, 26, 17, 75,
			69, 7, 66, 23};
		auto src = llo::Source<float>::get(mshape, data);

		ade::Shape mshape2({7, 2});
		std::vector<double> data2 = {71, 55, 48, 72, 43, 8, 71, 86, 43, 44, 25,
			50, 62, 66};
		auto src2 = llo::Source<double>::get(mshape2, data2);

		std::vector<uint16_t> data3 = {5, 27, 68, 92, 4, 3, 9, 20, 33, 65, 58, 36, 76, 78};
		auto src3 = llo::Source<uint16_t>::get(mshape2, data3);

		auto dest = llo::matmul(llo::div(src3, src2), llo::matmul(osrc, src));
		roots.push_back(dest);
	}

	// subtree three
	{
		ade::Shape mshape({3, 3});
		std::vector<int8_t> data = {6, -67, 0, -58, -62, -62, 5, -91, 76};
		auto src = llo::Source<int8_t>::get(mshape, data);

		std::vector<int64_t> data2 = {-36, 63, -68, 82, -5, 26, 89, -82, 79};
		auto src2 = llo::Source<int64_t>::get(mshape, data2);

		std::vector<int32_t> data3 = {-89, -91, -97, 4, -24, 71, 30, -44, 65};
		auto src3 = llo::Source<int32_t>::get(mshape, data3);

		auto dest = llo::sub(src, llo::prod({llo::abs(src), llo::exp(src2), llo::neg(src3)}));
		roots.push_back(dest);
	}

	save_graph(graph, roots);
	{
		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(graph.SerializeToOstream(&gotstr));
	}

	std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
	std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
	ASSERT_TRUE(expect_ifs.is_open());
	ASSERT_TRUE(got_ifs.is_open());

	std::string expect;
	std::string got;
	// skip the first line (it contains timestamp)
	expect_ifs >> expect;
	got_ifs >> got;
	for (size_t lineno = 1; expect_ifs && got_ifs; ++lineno)
	{
		expect_ifs >> expect;
		got_ifs >> got;
		EXPECT_STREQ(expect.c_str(), got.c_str()) << "line number " << lineno;
	}
}


#endif // DISABLE_SAVE_TEST
