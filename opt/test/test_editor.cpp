
#ifndef DISABLE_SHEAR_TEST


#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "opt/graph_edit.hpp"


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
		return shape_.to_string();
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

	double val_;

	ade::Shape shape_;
};


TEST(EDITOR, Prune)
{
	ade::TensptrT leaf(new MockTensor(ade::Shape()));
	ade::TensptrT leaf2(new MockTensor(ade::Shape()));
	ade::TensptrT leaf3(new MockTensor(ade::Shape()));
	ade::TensptrT mortal(ade::Functor::get(
		ade::Opcode{"killable", 0}, {ade::identity_map(leaf)}));
	ade::TensptrT mortal2(ade::Functor::get(
		ade::Opcode{"killable", 0}, {ade::identity_map(leaf2)}));
	ade::TensptrT immortal(ade::Functor::get(
		ade::Opcode{"not_killable", 2}, {ade::identity_map(leaf)}));
	ade::TensptrT binar(ade::Functor::get(
		ade::Opcode{"binary", 1}, {
			ade::identity_map(leaf3),
			ade::identity_map(mortal),
		}));
	ade::TensptrT binar2(ade::Functor::get(
		ade::Opcode{"binary", 1}, {
			ade::identity_map(mortal2),
			ade::identity_map(immortal),
		}));
	ade::TensptrT mortal3(ade::Functor::get(
		ade::Opcode{"killable", 0}, {ade::identity_map(binar)}));
	ade::TensptrT mortal4(ade::Functor::get(
		ade::Opcode{"killable", 0}, {ade::identity_map(mortal3)}));
	ade::TensptrT repl_binar(ade::Functor::get(
		ade::Opcode{"not_killable", 2}, {
			ade::identity_map(mortal4),
			ade::identity_map(binar2),
		}));

	opt::EditFuncT pruner =
		[&](bool& is_optimized,
			ade::Opcode& opcode, ade::ArgsT& args) -> ade::TensptrT
		{
			if (opcode.code_ < 2) // killable
			{
				ade::ArgsT filtered;
				for (auto arg : args)
				{
					ade::iTensor* tens = arg.get_tensor().get();
					if (tens != leaf.get() && tens != leaf2.get())
					{
						filtered.push_back(arg);
					}
				}
				args = filtered;
			}
			if (args.size() > 0)
			{
				is_optimized = true;
				return nullptr;
			}
			return leaf;
		};

	auto root = opt::graph_edit({repl_binar}, pruner)[0];

	std::unordered_map<ade::iTensor*,std::string> varlabels = {
		{leaf.get(), "leaf"},
		{leaf2.get(), "leaf2"},
		{leaf3.get(), "leaf3"},
	};
	std::stringstream str;
	str <<
		"(not_killable[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(killable[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(killable[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(binary[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(leaf3=[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(binary[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		"    `--(not_killable[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		"        `--(leaf=[1\\1\\1\\1\\1\\1\\1\\1])\n";
	EXPECT_STREQ("", compare_graph(str, root, true, varlabels).c_str());
}


#endif // DISABLE_SHEAR_TEST
