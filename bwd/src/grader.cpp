#include "bwd/grader.hpp"

#ifdef BWD_GRADER_HPP

namespace age
{

void Grader::visit (ade::iFunctor* func)
{
	const ade::Opcode sum = rules_->sum_opcode();
	const ade::Opcode prod = rules_->prod_opcode();
	if (func == target_)
	{
		derivatives_.emplace(func, rules_->data(1, target_->shape()));
		return;
	}

	ade::PathFinder finder(target_);
	func->accept(finder);

	auto& pathmap = finder.parents_;
	// no path to wrt
	if (pathmap.empty())
	{
		derivatives_.emplace(func, rules_->data(0, target_->shape()));
		return;
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	ade::GraphStat stat;
	func->accept(stat);

	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	parents.sort(
		[&](ade::iFunctor* a, ade::iFunctor* b)
		{
			return stat.graphsize_[a] > stat.graphsize_[b];
		});

	std::unordered_map<const ade::iTensor*,ade::TensT> grads = {
		{func, {rules_->data(1, func->shape())}},
	};
	for (ade::iFunctor* parent : parents)
	{
		ade::TensT& gargs = grads[parent];
		ade::TensptrT bwd = gargs.size() > 1 ? ade::TensptrT(
			ade::Functor::get(sum, to_args(gargs))) : gargs[0];

		auto& grad_indices = pathmap[parent];
		ade::ArgsT children = parent->get_children();
		size_t nchildren = children.size();
		// for each painted child, calculate dThis/dChild
		// go through grads in order
		std::list<size_t> ordered(grad_indices.begin(), grad_indices.end());
		ordered.sort();
		for (size_t i : ordered)
		{
			ade::TensT args;
			ade::MappedTensor& child = children[i];
			ade::CoordptrT revshaper(child.get_shaper()->reverse());
			bool revmapper = !child.map_io();
			ade::CoordptrT revcoorder = child.get_coorder();
			for (size_t j = 0; j < nchildren; ++j)
			{
				ade::MappedTensor& kid = children[j];
				ade::TensptrT tens = kid.get_tensor();
				if (j == i)
				{
					args.push_back(tens);
				}
				else
				{
					// reverse children[j] to child's shape/coord space
					args.push_back(ade::TensptrT(ade::Functor::get(sum, {
						ade::MappedTensor(
							ade::TensptrT(ade::Functor::get(sum, {kid})),
							revshaper,
							revmapper,
							revcoorder)
					})));
				}
			}
			// pass down forward-gradient pair
			ade::TensptrT grad(rules_->grad_rule(parent, args, i));

			ade::MappedTensor lhs(bwd, revshaper, revmapper, revcoorder);
			// avoid sum reduction for non-summation forward mapping
			// this is a hack to distinguish gradient for
			// generalized non-summation reduction and summation reduction
			// in the forward graph
			// e.g.:
			// PROD[1\2\3\1\1\1\1\1]) // sum every element sharing the same 3rd rank in var1 then multiple with var2
			// `--(SUM[1\2\3\1\1\1\1\1])
			// |   `--(var1([1\2\3\4\1\1\1\1]))
			// `--(var2([1\2\3\1\1\1\1\1]))
			// should be functionally different from
			// PROD[1\2\3\1\1\1\1\1]) // multiple the elements of the highest 3rd rank in each respective var
			// `--(var1([1\2\3\4\1\1\1\1]))
			// `--(var2([1\2\3\1\1\1\1\1]))
			// therefore its gradient should be different
			//
			// todo: find more a better way
			if (parent->get_opcode().code_ == sum.code_)
			{
				lhs = ade::identity_map(ade::TensptrT(
					ade::Functor::get(sum, {lhs})
				));
			}

			grads[child.get_tensor().get()].push_back(ade::TensptrT(
				ade::Functor::get(prod, {
					ade::identity_map(grad),
					lhs,
				})));
		}
	}
	auto finalgargs = grads[target_];
	derivatives_.emplace(func, finalgargs.size() > 1 ? ade::TensptrT(
		ade::Functor::get(sum, to_args(finalgargs))) : finalgargs[0]);
}

ade::ArgsT to_args (ade::TensT tens)
{
	ade::ArgsT args;
	std::transform(tens.begin(), tens.end(), std::back_inserter(args),
		[](ade::TensptrT& ten)
		{
			return ade::identity_map(ten);
		});
	return args;
}

}

#endif
