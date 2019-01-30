#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/opt/one_prune.hpp"

#include "llo/constant.hpp"

#ifdef LLO_ONE_PRUNE_HPP

namespace llo
{

ade::TensptrT one_prune_edit (bool& is_optimized,
	ade::Opcode& opcode, ade::ArgsT& args)
{
	size_t n = args.size();
	bool has_one = false;
	std::vector<bool> is_one(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto cst = dynamic_cast<llo::Constant*>(args[i].get_tensor().get());
		is_one[i] = nullptr != cst && 1 == cst->at<double>(0);
		has_one = has_one || is_one[i];
	}
	if (has_one)
	{
		switch (opcode.code_)
		{
			case age::ABS:
			case age::SQRT:
			case age::ROUND:
				return ade::TensptrT(llo::Constant::get(1, args[0].shape()));
			case age::LOG:
				return ade::TensptrT(llo::Constant::get(0, args[0].shape()));
			case age::POW:
				if (is_one[0])
				{
					return ade::TensptrT(llo::Constant::get(1, args[0].shape()));
				}
				// else if is_one[1]
				if (ade::identity == args[0].get_coorder())
				{
					return args[0].get_tensor();
				}
				is_optimized = true;
				opcode = ade::Opcode{"SUM", age::SUM};
				args = {args[0]};
				return nullptr;
			case age::PROD:
			{
				ade::ArgsT filtered;
				for (size_t i = 0, n = args.size(); i < n; ++i)
				{
					if (false == is_one[i])
					{
						filtered.push_back(args[i]);
					}
				}
				if (filtered.empty())
				{
					return ade::TensptrT(llo::Constant::get(1, args[0].shape()));
				}
				is_optimized = true;
				opcode = ade::Opcode{"PROD", age::PROD};
				args = filtered;
				return nullptr;
			}
			case age::DIV:
				if (is_one[1])
				{
					if (ade::identity == args[0].get_coorder())
					{
						return args[0].get_tensor();
					}
					is_optimized = true;
					opcode = ade::Opcode{"SUM", age::SUM};
					args = {args[0]};
					return nullptr;
				}
				// else if is_one[0]
				break;
			case age::NEG:
			case age::SIN:
			case age::COS:
			case age::TAN:
			case age::EXP:
			case age::SUM:
			case age::SUB:
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_UNIF:
			case age::RAND_NORM:
			case age::MATMUL:
				break;
			default:
				logs::fatalf("cannot one prune unknown opcode \"%s\"",
					opcode.name_.c_str());
		}
	}
	return nullptr;
}

ade::TensptrT one_prune (ade::TensptrT root)
{
	return opt::graph_edit(root, one_prune_edit);
}

}

#endif
