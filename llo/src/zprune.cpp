#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/zprune.hpp"

#ifdef LLO_ZPRUNE_HPP

namespace llo
{

// todo: move somewhere else
static ade::TensptrT prune0 (ade::iFunctor* func, ade::ArgsT args)
{
	age::_GENERATED_OPCODE opcode =
		(age::_GENERATED_OPCODE) func->get_opcode().code_;
	size_t n = args.size();
	bool has_zero = false;
	std::vector<bool> is_zero(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto var = dynamic_cast<llo::Variable*>(args[i].get_tensor().get());
		is_zero[i] = nullptr != var && "0" == var->label_;
		has_zero = has_zero || is_zero[i];
	}
	if (has_zero)
	{
		switch (opcode)
		{
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::TAN:
			case age::SQRT:
			case age::ROUND:
			case age::PROD:
				return ade::TensptrT(llo::get_scalar(0, func->shape()));
			case age::COS:
			case age::EXP:
				return ade::TensptrT(llo::get_scalar(1, func->shape()));
			case age::LOG:
				logs::fatal("cannot LOG by zero");
			case age::POW:
				if (is_zero[0])
				{
					return ade::TensptrT(llo::get_scalar(0, func->shape()));
				}
				// else if is_zero[1]
				return ade::TensptrT(llo::get_scalar(1, func->shape()));
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
					return ade::TensptrT(llo::get_scalar(0, func->shape()));
				}
				return ade::TensptrT(ade::Functor::get(ade::Opcode{"SUM", age::SUM}, filtered));
			}
			case age::SUB:
				if (is_zero[0] && is_zero[1])
				{
					return ade::TensptrT(llo::get_scalar(0, func->shape()));
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
				return ade::TensptrT(llo::get_scalar(0, func->shape()));
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_BINO:
			case age::RAND_UNIF:
			case age::RAND_NORM:
				break;
			default:
				logs::fatal("cannot zero prune unknown opcode");
		}
	}
	return nullptr;
}

ade::TensptrT zero_prune (ade::TensptrT root)
{
	opt::TargetedEdit zpruner(
		[](ade::iLeaf* leaf) -> bool
		{
			auto data = static_cast<Variable*>(leaf);
			if (nullptr == data)
			{
				return false;
			}
			return data->label_ == "0";
		}, prune0);
	return zpruner.prune(root);
}

ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target)
{
	age::Grader grader(target, std::make_shared<age::RuleSet>());
	root->accept(grader);
	auto it = grader.derivatives_.find(root.get());
	assert(grader.derivatives_.end() != it);
	return zero_prune(it->second);
}

}

#endif
