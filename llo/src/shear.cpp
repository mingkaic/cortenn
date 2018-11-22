#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/shear.hpp"

#ifdef LLO_SHEAR_HPP

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
				err::fatal("cannot LOG by zero");
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
					err::fatal("cannot DIV by zero");
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
				err::fatal("cannot prune unknown opcode");
		}
	}
	return ade::TensptrT(ade::Functor::get(ade::Opcode{age::name_op(opcode), opcode}, args));
}

ade::TensptrT zero_prune (ade::TensptrT root)
{
	// assert that context will be unaffected by prune,
	// since source will never be touched
	LabelFinder finder("0");
	root->accept(finder);
	auto& pathmap = finder.parents_;
	if (pathmap.empty()) // not path to zero or root is not a parent
	{
		return root;
	}
	ade::GraphStat stat;
	root->accept(stat);
	// grab the intersection of stat.funcs_ and pathmap
	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(), std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	parents.sort(
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	// each proceeding node in parents list is closer to SYMBOLIC_ZERO
	// start pruning according to each parent node in order
	std::unordered_set<ade::iTensor*> zeros = finder.labelled_;

	std::unordered_map<ade::iTensor*,ade::TensptrT> mapping;
	std::unordered_map<ade::iTensor*,bool> zeromap;
	for (ade::iFunctor* func : parents)
	{
		ade::ArgsT children = func->get_children();
		std::unordered_set<size_t> indices = pathmap[func]; // indices lead to zero nodes
		for (auto it = indices.begin(), et = indices.end(); it != et;)
		{
			ade::MappedTensor& child = children[*it];
			ade::iTensor* tens = child.tensor_.get();
			auto zit = zeros.find(tens);
			if (zeros.end() == zit) // child is not zero, so erase ot from indices
			{
				it = indices.erase(it);
			}
			else
			{
				++it;
			}
		}
		bool is_zero = false;
		mapping.emplace(func, prune0(is_zero, func, indices, children));
		if (is_zero) // func maps to zero, so store in zeros
		{
			zeros.emplace(func);
		}
	}
	auto it = mapping.find(root.get());
	if (mapping.end() == it)
	{
		err::fatal("something went wrong"); // todo: probably add context?
	}
	return it->second;
}

ade::TensptrT derive (ade::TensptrT root, ade::TensptrT target)
{
	age::Grader grader(target.get(), std::make_shared<age::RuleSet>());
	root->accept(grader);
	auto it = grader.derivatives_.find(root.get());
	assert(grader.derivatives_.end() != it);
	return zero_prune(it->second);
}

}

#endif
