
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "pbm/load.hpp"

#include "pbm/test/common.hpp"


const std::string testdir = "pbm/data";


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


TEST(LOAD, LoadGraph)
{
	tenncor::Graph graph;
	{
		std::fstream inputstr(testdir + "/graph.pb",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(graph.ParseFromIstream(&inputstr));
	}

	pbm::GraphInfo graphinfo;
	pbm::load_graph(graphinfo, graph,
		[](const char* pb, ade::Shape shape,
			size_t typecode, std::string label)
		{
			return ade::TensptrT(new MockTensor(shape));
		});

	EXPECT_EQ(2, graphinfo.roots_.size());

	ASSERT_EQ(3, graphinfo.tens_.children_.size());
	ASSERT_EQ(0, graphinfo.tens_.tens_.size());

	auto global_it = graphinfo.tens_.children_.find("global");
	auto subtree_it = graphinfo.tens_.children_.find("subtree");
	auto subtree2_it = graphinfo.tens_.children_.find("subtree2");

	ASSERT_NE(graphinfo.tens_.children_.end(), global_it) << "global namespace not found";
	ASSERT_NE(graphinfo.tens_.children_.end(), subtree_it) << "subtree namespace not found";
	ASSERT_NE(graphinfo.tens_.children_.end(), subtree2_it) << "subtree2 namespace not found";

	auto subtree = subtree_it->second;
	auto subtree2 = subtree2_it->second;
	ASSERT_EQ(3, subtree->tens_.size());
	ASSERT_EQ(4, subtree2->tens_.size());

	auto dest_it = subtree->tens_.find("dest");
	auto dest2_it = subtree2->tens_.find("dest");
	ASSERT_NE(subtree->tens_.end(), dest_it) << "{subtree, dest} not found";
	ASSERT_NE(subtree2->tens_.end(), dest2_it) << "{subtree2, dest} not found";

	ade::TensptrT tree1 = graphinfo.tens_.get_labelled({"subtree", "dest"});
	ade::TensptrT tree2 = graphinfo.tens_.get_labelled({"subtree2", "dest"});

	ASSERT_NE(nullptr, tree1);
	ASSERT_NE(nullptr, tree2);

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/graph.txt");
	ASSERT_TRUE(expectstr.is_open());
	while (std::getline(expectstr, line))
	{
		trim(line);
		if (line.size() > 0)
		{
			expect += line + '\n';
		}
	}

	PrettyEquation artist;
	std::stringstream gotstr;
	artist.print(gotstr, tree1);
	artist.print(gotstr, tree2);

#if 0
	std::cout << gotstr.str() << '\n';
#endif
	while (std::getline(gotstr, line))
	{
		trim(line);
		if (line.size() > 0)
		{
			got += line + '\n';
		}
	}

	EXPECT_STREQ(expect.c_str(), got.c_str());
}


#endif // DISABLE_LOAD_TEST
