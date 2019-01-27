#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/opt/zero_prune.hpp"

#ifdef LLO_ZERO_PRUNE_HPP

namespace llo
{

// todo: change this to target fixed value instead of looking at label
ade::TensptrT zero_prune_edit (ade::Opcode opcode, ade::ArgsT args)
{
	size_t n = args.size();
	bool has_zero = false;
	std::vector<bool> is_zero(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto var = dynamic_cast<llo::iVariable*>(args[i].get_tensor().get());
		is_zero[i] = nullptr != var && "0" == var->get_label();
		has_zero = has_zero || is_zero[i];
	}
	if (has_zero)
	{
		switch (opcode.code_)
		{
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::TAN:
			case age::SQRT:
			case age::ROUND:
			case age::PROD:
				return ade::TensptrT(llo::get_scalar(0, args[0].shape()));
			case age::COS:
			case age::EXP:
				return ade::TensptrT(llo::get_scalar(1, args[0].shape()));
			case age::LOG:
				logs::fatal("cannot LOG by zero");
			case age::POW:
				if (is_zero[0])
				{
					return ade::TensptrT(llo::get_scalar(0, args[0].shape()));
				}
				// else if is_zero[1]
				return ade::TensptrT(llo::get_scalar(1, args[1].shape()));
			case age::SUM:
			{
				ade::ArgsT filtered;
				for (size_t i = 0, n = args.size(); i < n; ++i)
				{
					if (false == is_zero[i])
					{
						filtered.push_back(args[i]);
					}
				}
				if (filtered.empty())
				{
					return ade::TensptrT(llo::get_scalar(0, args[0].shape()));
				}
				return ade::TensptrT(ade::Functor::get(ade::Opcode{"SUM", age::SUM}, filtered));
			}
			case age::SUB:
				if (is_zero[0] && is_zero[1])
				{
					return ade::TensptrT(llo::get_scalar(0, args[0].shape()));
				}
				else if (is_zero[0])
				{
					return ade::TensptrT(ade::Functor::get(ade::Opcode{"NEG", age::NEG}, {args[1]}));
				}
				// else if is_zero[1]
				return args[0].get_tensor();
			case age::DIV:
				if (is_zero[1])
				{
					logs::fatal("cannot DIV by zero");
				}
				// else if is_zero[0]
				return ade::TensptrT(llo::get_scalar(0, args[0].shape()));
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
				logs::fatalf("cannot zero prune unknown opcode \"%s\"",
					opcode.name_.c_str());
		}
	}
	return nullptr;
}

ade::TensptrT zero_prune (ade::TensptrT root)
{
	return opt::graph_edit(root, zero_prune_edit);
}

}

#endif
