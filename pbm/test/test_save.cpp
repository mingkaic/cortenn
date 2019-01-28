
#ifndef DISABLE_SAVE_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "pbm/save.hpp"

#include "pbm/test/common.hpp"


const std::string testdir = "pbm/data";


TEST(SAVE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/graph.pb";
	std::string got_pbfile = "got_graph.pb";
	cortenn::Graph graph;
	std::vector<ade::TensptrT> roots;

	pbm::PathedMapT labels;
	// subtree one
	ade::Shape shape({3, 7});
	ade::TensptrT osrc(new MockTensor(shape));

	ade::Shape shape2({7, 3});
	ade::TensptrT osrc2(new MockTensor(shape2));

	labels[osrc] = {"global", "osrc"};
	labels[osrc2] = {"global", "osrc2"};

	{
		ade::TensptrT src(new MockTensor(shape));

		ade::Shape shape3({3, 1, 7});
		ade::TensptrT src2(new MockTensor(shape3));

		ade::TensptrT dest(ade::Functor::get(ade::Opcode{"-", 0}, {
			{src2, ade::identity},
			{ade::TensptrT(ade::Functor::get(ade::Opcode{"@", 1}, {
				{ade::TensptrT(ade::Functor::get(ade::Opcode{"/", 2}, {
					{ade::TensptrT(ade::Functor::get(ade::Opcode{"neg", 3}, {
						{osrc, ade::identity},
					})), ade::identity},
					{ade::TensptrT(ade::Functor::get(ade::Opcode{"+", 4}, {
						{ade::TensptrT(
							ade::Functor::get(ade::Opcode{"sin", 5}, {
							{src, ade::identity}})), ade::identity},
						{src, ade::identity},
					})), ade::identity}
				})), ade::permute({1, 0})},
				{osrc2, ade::identity}
			})), ade::permute({1, 2, 0})},
		}));
		roots.push_back(dest);

		labels[src] = {"subtree", "src"};
		labels[src2] = {"subtree", "src2"};
		labels[dest] = {"subtree", "dest"};
	}

	// subtree two
	{
		ade::Shape mshape({3, 3});
		ade::TensptrT src(new MockTensor(mshape));

		ade::TensptrT src2(new MockTensor(mshape));

		ade::TensptrT src3(new MockTensor(mshape));

		ade::TensptrT dest(ade::Functor::get(ade::Opcode{"-", 0}, {
			{src, ade::identity},
			{ade::TensptrT(ade::Functor::get(ade::Opcode{"*", 6}, {
				{ade::TensptrT(ade::Functor::get(ade::Opcode{"abs", 7}, {
					{src, ade::identity},
				})), ade::identity},
				{ade::TensptrT(ade::Functor::get(ade::Opcode{"exp", 8}, {
					{src2, ade::identity},
				})), ade::identity},
				{ade::TensptrT(ade::Functor::get(ade::Opcode{"neg", 3}, {
					{src3, ade::identity},
				})), ade::identity},
			})), ade::identity},
		}));
		roots.push_back(dest);

		labels[src] = {"subtree2", "src"};
		labels[src2] = {"subtree2", "src2"};
		labels[src3] = {"subtree2", "src3"};
		labels[dest] = {"subtree2", "dest"};
	}

	pbm::GraphSaver saver(
		[](bool& is_const, ade::iLeaf* leaf)
		{
			return std::string(leaf->shape().n_elems(), 0);
		});
	for (auto& root : roots)
	{
		root->accept(saver);
	}

	saver.save(graph, labels);

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
