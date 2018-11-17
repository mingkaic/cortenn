
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "pbm/graph.hpp"


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

	std::vector<llo::Variable> roots;
	{
		std::vector<llo::Variable> nodes = load_graph(graph);
		std::unordered_set<ade::iTensor*> have_parents;
		for (auto it = nodes.begin(), et = nodes.end(); et != it; ++it)
		{
			if (ade::iFunctor* func =
				dynamic_cast<ade::iFunctor*>(it->tensor_.get()))
			{
				ade::ArgsT refs = func->get_children();
				for (auto& ref : refs)
				{
					have_parents.emplace(ref.tensor_.get());
				}
			}
		}

		// filter out nodes with parents to find roots
		std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(roots),
		[&](llo::Variable& node)
		{
			return have_parents.end() == have_parents.find(node.tensor_.get());
		});
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
			expect += line + "\n";
		}
	}
	for (llo::Variable& root : roots)
	{
		PrettyEquation artist;
		std::stringstream gotstr;
		artist.print(gotstr, root.tensor_);

#if 0
		std::cout << gotstr.str() << std::endl;
#endif
		while (std::getline(gotstr, line))
		{
			trim(line);
			if (line.size() > 0)
			{
				got += line + "\n";
			}
		}
	}
	EXPECT_STREQ(expect.c_str(), got.c_str());
}


#endif // DISABLE_LOAD_TEST
