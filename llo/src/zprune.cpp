#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/zprune.hpp"

#ifdef LLO_ZPRUNE_HPP

namespace llo
{

// todo: move somewhere else
static ade::TensptrT prune0 (bool& is_zero, ade::iFunctor* func,
	std::unordered_set<size_t> zeros, ade::ArgsT args)
{
	is_zero = false;
	age::_GENERATED_OPCODE opcode =
		(age::_GENERATED_OPCODE) func->get_opcode().code_;
	if (false == zeros.empty())
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
				is_zero = true;
				return ade::TensptrT(age::data(0, func->shape()));
			case age::COS:
			case age::EXP:
				return ade::TensptrT(age::data(1, func->shape()));
			case age::LOG:
				logs::fatal("cannot LOG by zero");
			case age::POW:
				if (zeros.end() != zeros.find(0))
				{
					is_zero = true;
					return ade::TensptrT(age::data(0, func->shape()));
				}
				// else if zeros.end() != zeros.find(1)
				return ade::TensptrT(age::data(1, func->shape()));
			case age::SUM:
			{
				ade::ArgsT filtered;
				for (size_t i = 0, n = args.size(); i < n; ++i)
				{
					if (zeros.end() == zeros.find(i))
					{
						filtered.push_back(args[i]);
					}
				}
				if (filtered.empty())
				{
					is_zero = true;
					return ade::TensptrT(age::data(0, func->shape()));
				}
				return ade::TensptrT(ade::Functor::get(ade::Opcode{"SUM", age::SUM}, filtered));
			}
			case age::SUB:
				if (2 == zeros.size())
				{
					is_zero = true;
					return ade::TensptrT(age::data(0, func->shape()));
				}
				else if (zeros.end() != zeros.find(0))
				{
					return ade::TensptrT(ade::Functor::get(ade::Opcode{"NEG", age::NEG}, {args[1]}));
				}
				// else if zeros.end() != zeros.find(1)
				return args[0].tensor_;
			case age::DIV:
				if (zeros.end() != zeros.find(1))
				{
					logs::fatal("cannot DIV by zero");
				}
				// else if 0 == zeros.front()
				is_zero = true;
				return ade::TensptrT(age::data(0, func->shape()));
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
				logs::fatal("cannot prune unknown opcode");
		}
	}
	return ade::TensptrT(ade::Functor::get(ade::Opcode{age::name_op(opcode), opcode}, args));
}

ade::TensptrT zero_prune (ade::TensptrT root)
{
	opt::TargetPruner<std::string> zpruner("0",
		[](ade::iLeaf* leaf)
		{
			auto data = static_cast<Variable*>(leaf);
			return data->label_;
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
