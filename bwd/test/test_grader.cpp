
#ifndef DISABLE_GRADER_TEST


#include <sstream>

#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "exam/exam.hpp"

#include "bwd/grader.hpp"


struct MockTensor final : public ade::iLeaf
{
	MockTensor (void) = default;

	MockTensor (ade::Shape shape) : shape_(shape) {}

	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return "MockTensor";
	}

	void* data (void) override
	{
		return &val_;
	}

	const void* data (void) const override
	{
		return &val_;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	std::string type_label (void) const override
	{
		return "";
	}

	size_t nbytes (void) const override
	{
		return 0;
	}

	double val_;

	ade::Shape shape_;
};


struct MockRuleSet final : public age::iRuleSet
{
	ade::LeafptrT data (double scalar, ade::Shape shape) override
	{
		auto out = new ::MockTensor(shape);
		out->val_ = scalar;
		return ade::LeafptrT(out);
	}

	ade::Opcode sum_opcode (void) override
	{
		return ade::Opcode{"+", 0};
	}

	ade::TensptrT chain_rule (ade::iFunctor* fwd,
		ade::FuncArg bwd, ade::TensT args, size_t idx) override
	{
		ade::Opcode outcode;
		ade::Opcode fwd_opcode = fwd->get_opcode();
		// grad of sum is prod and grad of prod is sum
		if (fwd_opcode.code_)
		{
			outcode = sum_opcode();
		}
		else
		{
			outcode = prod_opcode();
		}
		return mul(ade::TensptrT(ade::Functor::get(outcode, ade::to_args(args))),
			ade::TensptrT(ade::Functor::get(fwd_opcode, {bwd})));
	}

	ade::Opcode prod_opcode (void)
	{
		return ade::Opcode{"*", 1};
	}

	ade::TensptrT mul (ade::TensptrT a, ade::TensptrT b)
	{
		return ade::TensptrT(ade::Functor::get(prod_opcode(), {
			ade::identity_map(a), ade::identity_map(b),
		}));
	}
};


static std::shared_ptr<MockRuleSet> mock_rules =
	std::make_shared<MockRuleSet>();


ade::TensptrT derive (ade::TensptrT& root, const ade::iTensor* wrt)
{
	age::Grader grader(wrt, mock_rules);
	root->accept(grader);
	auto it = grader.derivatives_.find(root.get());
	assert(grader.derivatives_.end() != it);
	return it->second;
}


#define COORD_EQ(expect, got)\
{\
	expect->access([&](const ade::MatrixT& expectm)\
	{\
		got->access([&](const ade::MatrixT& gotm)\
		{\
			for (size_t i = 0; i < ade::mat_dim; ++i)\
			{\
				for (size_t j = 0; j < ade::mat_dim; ++j)\
				{\
					EXPECT_EQ(expectm[i][j], gotm[i][j]) <<\
						"coord(" << i << "," << j << ")";\
				}\
			}\
		});\
	});\
}


#define ARR_EQ(expect, gbegin, gend)\
EXPECT_TRUE(std::equal(expect.begin(), expect.end(), gbegin)) <<\
	fmts::to_string(expect.begin(), expect.end()) << " not equal to " <<\
	fmts::to_string(gbegin, gend);


TEST(GRADER, Ruleset)
{
	ade::TensptrT tens(new MockTensor());

	EXPECT_FATAL(age::Grader(nullptr, mock_rules), "cannot derive with respect to null");
	EXPECT_FATAL(age::Grader(tens.get(), nullptr), "cannot derive without ruleset");
}


TEST(GRADER, Leaf)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT g1(derive(leaf, leaf.get()));
	ade::TensptrT g0(derive(leaf, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream sstr;
	sstr << "(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	EXPECT_STREQ("", compare_graph(sstr, g1).c_str());
	sstr.clear();
	sstr << "(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	EXPECT_STREQ("", compare_graph(sstr, g0).c_str());;
}


TEST(GRADER, Sum)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT fwd(ade::Functor::get(
		mock_rules->sum_opcode(), {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	zstr << "(MockTensor[7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" << // chain rule (derivative of SUM is PROD)
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" << // chain rule
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(ostr, g1).c_str());;
	EXPECT_STREQ("", compare_graph(zstr, g0).c_str());;
	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;
}


TEST(GRADER, Prod)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT fwd(ade::Functor::get(
		mock_rules->prod_opcode(), {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	zstr << "(MockTensor[7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" << // chain rule (derivative of PROD is SUM)
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" << // chain rule
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(ostr, g1).c_str());;
	EXPECT_STREQ("", compare_graph(zstr, g0).c_str());;
	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;
}


TEST(GRADER, SumProd)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT prod(ade::Functor::get(
		mock_rules->prod_opcode(), {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	ade::TensptrT sum(ade::Functor::get(
		mock_rules->sum_opcode(), {
		ade::identity_map(prod),
		ade::identity_map(prod),
	}));

	ade::TensptrT gl(derive(sum, leaf.get()));
	ade::TensptrT gr(derive(sum, leaf1.get()));

	std::stringstream lstr;
	std::stringstream rstr;

	lstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |       `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |           `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |               `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |               `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |       `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"                 `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |       `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |           `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |               `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   |               `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"         `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |       `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |   `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             |       `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"             `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"                 `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;
}


TEST(GRADER, Extend)
{
	std::vector<ade::DimT> slist = {2};
	std::vector<ade::DimT> slist1 = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist1)));

	auto left = ade::extend_map(leaf, 1, {3, 4});
	auto right = ade::extend_map(leaf1, 2, {4});
	ade::TensptrT fwd(
		ade::Functor::get(mock_rules->sum_opcode(), {left, right}));

	ade::CoordptrT leftmapper = left.get_shaper();
	ade::CoordptrT rightmapper = right.get_shaper();
	ade::CoordptrT leftrev(leftmapper->reverse());
	ade::CoordptrT rightrev(rightmapper->reverse());

	COORD_EQ(leftrev, left.get_coorder());
	COORD_EQ(rightrev, right.get_coorder());

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n";
	zstr << "(MockTensor[7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(*[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\1\\1\\1\\1\\1\\1\\1])\n" << // chain rule (derivative of SUM is PROD)
		" |   `--(MockTensor[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(+[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\1\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\1\\1\\1\\1\\1\\1])\n" << // chain rule
		" |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(ostr, g1).c_str());;
	EXPECT_STREQ("", compare_graph(zstr, g0).c_str());;
	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;

	{
		auto fl = dynamic_cast<ade::Functor*>(gl.get());
		ASSERT_NE(nullptr, fl);

		auto children = fl->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(1, gchildren.size());
			EXPECT_TRUE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectout{2,1,1,1,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				// shaper is always input to output
				ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectout, out, out + ade::rank_cap);
				// simulate out to input
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectout, out, out + ade::rank_cap);
			}
			COORD_EQ(leftrev, target_shaper);
			COORD_EQ(leftrev, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(
					gchildren[1].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_TRUE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectout{2,1,1,1,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectout, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectout, out, out + ade::rank_cap);
				}
				COORD_EQ(leftrev, target_shaper);
				COORD_EQ(leftrev, target_mapper);
			}
		}
	}

	{
		auto fr = dynamic_cast<ade::Functor*>(gr.get());
		ASSERT_NE(nullptr, fr);

		auto children = fr->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(1, gchildren.size());
			EXPECT_TRUE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectout{2,3,1,1,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectout, out, out + ade::rank_cap);
				// simulate input to output
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectout, out, out + ade::rank_cap);
			}
			COORD_EQ(rightrev, target_shaper);
			COORD_EQ(rightrev, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[0].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_TRUE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectout{2,3,1,1,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectout, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectout, out, out + ade::rank_cap);
				}
				COORD_EQ(rightrev, target_shaper);
				COORD_EQ(rightrev, target_mapper);
			}
		}
	}
}


TEST(GRADER, ReduceExtend)
{
	std::vector<ade::DimT> slist = {2, 3, 4};
	std::vector<ade::DimT> slist1 = {2};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist1)));

	auto left = ade::reduce_map(leaf, 2, {4});
	auto right = ade::extend_map(leaf1, 1, {3});
	ade::TensptrT fwd(
		ade::Functor::get(mock_rules->sum_opcode(), {left, right}));

	ade::CoordptrT leftmapper = left.get_shaper();
	ade::CoordptrT rightmapper = right.get_shaper();
	ade::CoordptrT leftrev(leftmapper->reverse());
	ade::CoordptrT rightrev(rightmapper->reverse());

	COORD_EQ(leftmapper, left.get_coorder());
	COORD_EQ(rightrev, right.get_coorder());

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	zstr << "(MockTensor[7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(*[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\4\\1\\1\\1\\1\\1])\n" << // chain rule (derivative of SUM is PROD)
		" |   `--(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |       `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\4\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(*[2\\1\\1\\1\\1\\1\\1\\1])\n" << // chain rule
		" |   `--(+[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(+[2\\1\\1\\1\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\1\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(ostr, g1).c_str());;
	EXPECT_STREQ("", compare_graph(zstr, g0).c_str());;
	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;

	{
		auto fl = dynamic_cast<ade::Functor*>(gl.get());
		ASSERT_NE(nullptr, fl);

		auto children = fl->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_FALSE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectshape{2,3,4,1,1,1,1,1};
				std::vector<double> expectcoord{2,3,1,1,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				// shaper is always input to output
				ade::CDimT cin[ade::rank_cap] = {2,3,1,1,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectshape, out, out + ade::rank_cap);
				// simulate out to input
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectcoord, out, out + ade::rank_cap);
			}
			COORD_EQ(leftrev, target_shaper);
			COORD_EQ(leftmapper, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, children.size());
			EXPECT_EQ(ade::identity, children[0].get_shaper());
			EXPECT_EQ(ade::identity, children[0].get_coorder());
			EXPECT_EQ(ade::identity, children[1].get_shaper());
			EXPECT_EQ(ade::identity, children[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[1].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_FALSE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectshape{2,3,4,1,1,1,1,1};
					std::vector<double> expectcoord{2,3,1,1,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {2,3,1,1,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectshape, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectcoord, out, out + ade::rank_cap);
				}
				COORD_EQ(leftrev, target_shaper);
				COORD_EQ(leftmapper, target_mapper);

			}
		}
	}

	{
		auto fr = dynamic_cast<ade::Functor*>(gr.get());
		ASSERT_NE(nullptr, fr);

		auto children = fr->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_TRUE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectout{2,1,1,1,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				ade::CDimT cin[ade::rank_cap] = {2,3,1,1,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectout, out, out + ade::rank_cap);
				// simulate input to output
				ade::CDimT cin2[ade::rank_cap] = {2,3,1,1,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectout, out, out + ade::rank_cap);
			}
			COORD_EQ(rightrev, target_shaper);
			COORD_EQ(rightrev, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[0].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_TRUE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectout{2,1,1,1,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					ade::CDimT cin[ade::rank_cap] = {2,3,1,1,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectout, out, out + ade::rank_cap);
					// simulate input to output
					ade::CDimT cin2[ade::rank_cap] = {2,3,1,1,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectout, out, out + ade::rank_cap);
				}
				COORD_EQ(rightrev, target_shaper);
				COORD_EQ(rightrev, target_mapper);
			}
		}
	}
}


TEST(GRADER, PermuteReduce)
{
	std::vector<ade::DimT> slist = {4, 2, 1, 3};
	std::vector<ade::DimT> slist1 = {2, 3, 4, 5};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist1)));

	auto left = ade::permute_map(leaf, {1, 3, 0});
	auto right = ade::reduce_map(leaf1, 3, {5});
	ade::TensptrT fwd(
		ade::Functor::get(mock_rules->sum_opcode(), {left, right}));

	ade::CoordptrT leftmapper = left.get_shaper();
	ade::CoordptrT rightmapper = right.get_shaper();
	ade::CoordptrT leftrev(leftmapper->reverse());
	ade::CoordptrT rightrev(rightmapper->reverse());

	COORD_EQ(leftrev, left.get_coorder());
	COORD_EQ(rightmapper, right.get_coorder());

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n";
	zstr << "(MockTensor[7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(*[4\\2\\1\\3\\1\\1\\1\\1])\n" <<
		" `--(*[4\\2\\1\\3\\1\\1\\1\\1])\n" << // chain rule (derivative of SUM is PROD)
		" |   `--(MockTensor[4\\2\\1\\3\\1\\1\\1\\1])\n" <<
		" |   `--(+[4\\2\\1\\3\\1\\1\\1\\1])\n" <<
		" |       `--(+[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[2\\3\\4\\5\\1\\1\\1\\1])\n" <<
		" `--(+[4\\2\\1\\3\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[2\\3\\4\\5\\1\\1\\1\\1])\n" <<
		" `--(*[2\\3\\4\\5\\1\\1\\1\\1])\n" << // chain rule
		" |   `--(+[2\\3\\4\\5\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[2\\3\\4\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[4\\2\\1\\3\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[2\\3\\4\\5\\1\\1\\1\\1])\n" <<
		" `--(+[2\\3\\4\\5\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[2\\3\\4\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(ostr, g1).c_str());;
	EXPECT_STREQ("", compare_graph(zstr, g0).c_str());;
	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;

	{
		auto fl = dynamic_cast<ade::Functor*>(gl.get());
		ASSERT_NE(nullptr, fl);

		auto children = fl->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_TRUE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectout{4,2,1,3,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				// shaper is always input to output
				ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectout, out, out + ade::rank_cap);
				// simulate out to input
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectout, out, out + ade::rank_cap);
			}
			COORD_EQ(leftrev, target_shaper);
			COORD_EQ(leftrev, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[1].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_TRUE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectout{4,2,1,3,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectout, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectout, out, out + ade::rank_cap);
				}
				COORD_EQ(leftrev, target_shaper);
				COORD_EQ(leftrev, target_mapper);
			}
		}
	}

	{
		auto fr = dynamic_cast<ade::Functor*>(gr.get());
		ASSERT_NE(nullptr, fr);

		auto children = fr->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_FALSE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectshape{2,3,4,5,1,1,1,1};
				std::vector<double> expectcoord{2,3,4,1,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectshape, out, out + ade::rank_cap);
				// simulate input to output
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,5,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectcoord, out, out + ade::rank_cap);
			}
			COORD_EQ(rightrev, target_shaper);
			COORD_EQ(rightmapper, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[0].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_FALSE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectshape{2,3,4,5,1,1,1,1};
					std::vector<double> expectcoord{2,3,4,1,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {2,3,4,1,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectshape, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,5,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectcoord, out, out + ade::rank_cap);
				}
				COORD_EQ(rightrev, target_shaper);
				COORD_EQ(rightmapper, target_mapper);
			}
		}
	}
}


TEST(GRADER, DiffShaperCoorder)
{
	std::vector<ade::DimT> slist = {4, 4, 3, 3};
	std::vector<ade::DimT> slist1 = {3, 3, 4, 4};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist1)));

	ade::CoordptrT leftshaper = ade::permute({2, 1, 3, 0});
	ade::CoordptrT rightshaper = ade::permute({0, 3, 1, 2});
	ade::CoordptrT leftmapper = ade::permute({1, 3, 0, 2});
	ade::CoordptrT rightmapper = ade::permute({1, 2, 0, 3});
	ade::FuncArg left(leaf, leftshaper, false, leftmapper);
	ade::FuncArg right(leaf1, rightshaper, true, rightmapper);
	ade::TensptrT fwd(
		ade::Functor::get(mock_rules->sum_opcode(), {left, right}));

	ade::CoordptrT leftshaperev(leftshaper->reverse());
	ade::CoordptrT rightshaperev(rightshaper->reverse());
	ade::CoordptrT leftcoordrev(leftmapper->reverse());
	ade::CoordptrT rightcoordrev(rightmapper->reverse());

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	ASSERT_NE(nullptr, mock1);
	ASSERT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "(MockTensor[3\\4\\3\\4\\1\\1\\1\\1])\n";
	zstr << "(MockTensor[7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(*[4\\4\\3\\3\\1\\1\\1\\1])\n" <<
		" `--(*[4\\4\\3\\3\\1\\1\\1\\1])\n" << // chain rule (derivative of SUM is PROD)
		" |   `--(MockTensor[4\\4\\3\\3\\1\\1\\1\\1])\n" <<
		" |   `--(+[4\\4\\3\\3\\1\\1\\1\\1])\n" <<
		" |       `--(+[3\\4\\3\\4\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[3\\3\\4\\4\\1\\1\\1\\1])\n" <<
		" `--(+[4\\4\\3\\3\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[3\\4\\3\\4\\1\\1\\1\\1])\n";
	rstr <<
		"(*[3\\3\\4\\4\\1\\1\\1\\1])\n" <<
		" `--(*[3\\3\\4\\4\\1\\1\\1\\1])\n" << // chain rule
		" |   `--(+[3\\3\\4\\4\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[3\\4\\3\\4\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[4\\4\\3\\3\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[3\\3\\4\\4\\1\\1\\1\\1])\n" <<
		" `--(+[3\\3\\4\\4\\1\\1\\1\\1])\n" << // derivative of leaf wrt leaf
		"     `--(MockTensor[3\\4\\3\\4\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(ostr, g1).c_str());;
	EXPECT_STREQ("", compare_graph(zstr, g0).c_str());;
	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;

	{
		auto fl = dynamic_cast<ade::Functor*>(gl.get());
		ASSERT_NE(nullptr, fl);

		auto children = fl->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_TRUE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectshape{4,4,3,3,1,1,1,1};
				std::vector<double> expectcoord{3,5,2,4,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				// shaper is always input to output
				ade::CDimT cin[ade::rank_cap] = {3,4,3,4,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectshape, out, out + ade::rank_cap);
				// simulate out to input
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,5,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectcoord, out, out + ade::rank_cap);
			}
			COORD_EQ(leftshaperev, target_shaper);
			COORD_EQ(leftmapper, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[1].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_TRUE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectshape{4,4,3,3,1,1,1,1};
					std::vector<double> expectcoord{3,5,2,4,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {3,4,3,4,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectshape, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,5,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectcoord, out, out + ade::rank_cap);
				}
				COORD_EQ(leftshaperev, target_shaper);
				COORD_EQ(leftmapper, target_mapper);
			}
		}
	}

	{
		auto fr = dynamic_cast<ade::Functor*>(gr.get());
		ASSERT_NE(nullptr, fr);

		auto children = fr->get_children();
		EXPECT_EQ(2, children.size());
		EXPECT_EQ(ade::identity, children[0].get_shaper());
		EXPECT_EQ(ade::identity, children[0].get_coorder());
		EXPECT_EQ(ade::identity, children[1].get_shaper());
		EXPECT_EQ(ade::identity, children[1].get_coorder());

		{
			auto child = dynamic_cast<ade::Functor*>(
				children[1].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_FALSE(gchildren[0].map_io());
			auto target_shaper = gchildren[0].get_shaper();
			auto target_mapper = gchildren[0].get_coorder();
			{
				std::vector<double> expectshape{3,3,4,4,1,1,1,1};
				std::vector<double> expectcoord{3,4,2,5,1,1,1,1};
				ade::CDimT out[ade::rank_cap];
				ade::CDimT cin[ade::rank_cap] = {3,4,3,4,1,1,1,1};
				target_shaper->forward(out, cin);
				ARR_EQ(expectshape, out, out + ade::rank_cap);
				// simulate input to output
				ade::CDimT cin2[ade::rank_cap] = {2,3,4,5,1,1,1,1};
				target_mapper->forward(out, cin2);
				ARR_EQ(expectcoord, out, out + ade::rank_cap);
			}
			COORD_EQ(rightshaperev, target_shaper);
			COORD_EQ(rightmapper, target_mapper);
		}

		{
			auto child = dynamic_cast<ade::Functor*>(children[0].get_tensor().get());
			ASSERT_NE(nullptr, child);

			auto gchildren = child->get_children();
			EXPECT_EQ(2, gchildren.size());
			EXPECT_EQ(ade::identity, gchildren[0].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[0].get_coorder());
			EXPECT_EQ(ade::identity, gchildren[1].get_shaper());
			EXPECT_EQ(ade::identity, gchildren[1].get_coorder());

			{
				auto gchild = dynamic_cast<ade::Functor*>(gchildren[0].get_tensor().get());
				ASSERT_NE(nullptr, gchild);

				auto ggchildren = gchild->get_children();
				EXPECT_EQ(1, ggchildren.size());
				EXPECT_FALSE(ggchildren[0].map_io());
				auto target_shaper = ggchildren[0].get_shaper();
				auto target_mapper = ggchildren[0].get_coorder();
				{
					std::vector<double> expectshape{3,3,4,4,1,1,1,1};
					std::vector<double> expectcoord{3,4,2,5,1,1,1,1};
					ade::CDimT out[ade::rank_cap];
					// shaper is always input to output
					ade::CDimT cin[ade::rank_cap] = {3,4,3,4,1,1,1,1};
					target_shaper->forward(out, cin);
					ARR_EQ(expectshape, out, out + ade::rank_cap);
					// simulate out to input
					ade::CDimT cin2[ade::rank_cap] = {2,3,4,5,1,1,1,1};
					target_mapper->forward(out, cin2);
					ARR_EQ(expectcoord, out, out + ade::rank_cap);
				}
				COORD_EQ(rightshaperev, target_shaper);
				COORD_EQ(rightmapper, target_mapper);
			}
		}
	}
}


TEST(GRADER, NoMapAliasing)
{
	std::vector<ade::DimT> slist = {1, 2, 3, 4};
	std::vector<ade::DimT> slist1 = {1, 2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist1)));

	ade::FuncArg left = reduce_map(leaf, 3, {4});
	ade::FuncArg right = ade::identity_map(leaf1);
	ade::TensptrT fwd(
		ade::Functor::get(mock_rules->prod_opcode(), {left, right}));
	ade::TensptrT fwd2(
		ade::Functor::get(mock_rules->prod_opcode(), {
			ade::identity_map(ade::TensptrT(ade::Functor::get(
				mock_rules->sum_opcode(), {left}))), right}));

	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));
	ade::TensptrT gl2(derive(fwd2, leaf.get()));
	ade::TensptrT gr2(derive(fwd2, leaf1.get()));

	std::stringstream lstr;
	std::stringstream rstr;
	std::stringstream lstr2;
	std::stringstream rstr2;

	lstr <<
		"(*[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" `--(+[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" |   `--(+[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" |       `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |           `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" `--(*[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		"     `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(*[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MockTensor[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" `--(*[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"     `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n";

	lstr2 <<
		"(*[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" `--(*[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" `--(+[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		"     `--(*[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"         `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"         |   `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"         |   |   `--(MockTensor[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		"         |   `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"         |       `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"         |           `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"         `--(*[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"             `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n";
	rstr2 <<
		"(*[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(+[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" |   |           `--(MockTensor[1\\2\\3\\4\\1\\1\\1\\1])\n" <<
		" |   `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		" `--(*[1\\2\\3\\1\\1\\1\\1\\1])\n" <<
		"     `--(MockTensor[1\\2\\3\\1\\1\\1\\1\\1])\n";

	EXPECT_STREQ("", compare_graph(lstr, gl).c_str());;
	EXPECT_STREQ("", compare_graph(rstr, gr).c_str());;
	EXPECT_STREQ("", compare_graph(lstr2, gl2).c_str());;
	EXPECT_STREQ("", compare_graph(rstr2, gr2).c_str());;
}


#endif // DISABLE_GRADER_TEST
