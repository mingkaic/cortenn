#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/opt/one_prune.hpp"

#ifdef LLO_ONE_PRUNE_HPP

namespace llo
{

// todo: change this to target fixed value instead of looking at label
ade::TensptrT one_prune_edit (ade::Opcode opcode, ade::ArgsT args)
{
	size_t n = args.size();
	bool has_one = false;
	std::vector<bool> is_one(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto var = dynamic_cast<llo::iVariable*>(args[i].get_tensor().get());
		is_one[i] = nullptr != var && "1" == var->get_label();
		has_one = has_one || is_one[i];
	}
	if (has_one)
	{
		switch (opcode.code_)
		{
			case age::ABS:
			case age::SQRT:
			case age::ROUND:
				return ade::TensptrT(llo::get_scalar(1, args[0].shape()));
			case age::LOG:
				return ade::TensptrT(llo::get_scalar(0, args[0].shape()));
			case age::POW:
				if (is_one[0])
				{
					return ade::TensptrT(llo::get_scalar(1, args[0].shape()));
				}
				// else if is_one[1]
				if (ade::identity == args[0].get_coorder())
				{
					return args[0].get_tensor();
				}
				return ade::TensptrT(ade::Functor::get(
					ade::Opcode{"SUM", age::SUM}, {args[0]}));
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
					return ade::TensptrT(llo::get_scalar(1, args[0].shape()));
				}
				return ade::TensptrT(ade::Functor::get(
					ade::Opcode{"PROD", age::PROD}, filtered));
			}
			case age::DIV:
				if (is_one[1])
				{
					if (ade::identity == args[0].get_coorder())
					{
						return args[0].get_tensor();
					}
					return ade::TensptrT(ade::Functor::get(
						ade::Opcode{"SUM", age::SUM}, {args[0]}));
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
