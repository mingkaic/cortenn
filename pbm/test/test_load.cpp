
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "pbm/load.hpp"

#include "pbm/test/common.hpp"


struct MockLoader : public pbm::iDataLoader
{
    virtual ~MockLoader (void) = default;

    virtual ade::TensptrT deserialize (const char* pb,
        ade::Shape shape, size_t typecode, std::string label)
	{
		return ade::TensptrT(new MockTensor(shape));
	}
};


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

	MockLoader loader;
	pbm::GraphInfo graphinfo;
	pbm::load_graph(graphinfo, graph, loader);

	EXPECT_EQ(2, graphinfo.roots_.size());

	ade::TensptrT tree1, tree2;
	for (auto& lvar : graphinfo.labelled_)
	{
		std::string treeid = lvar.second.front();
		std::string instid = lvar.second.back();

		if (instid == "dest")
		{
			if (treeid == "subtree")
			{
				tree1 = lvar.first;
			}
			else if (treeid == "subtree2")
			{
				tree2 = lvar.first;
			}
		}
	}

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
