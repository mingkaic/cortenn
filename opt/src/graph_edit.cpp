#include <list>

#include "opt/graph_edit.hpp"

#ifdef OPT_SHEAR_HPP

namespace opt
{

ade::TensptrT graph_edit (ade::TensptrT root, EditFuncT edit)
{
	ade::GraphStat stat;
	root->accept(stat);
	if (stat.graphsize_.size() == 0)
	{
		return root;
	}
	std::unordered_map<ade::iTensor*,size_t> funcsize;
	std::copy_if(stat.graphsize_.begin(), stat.graphsize_.end(),
		std::inserter(funcsize, funcsize.end()),
		[](std::pair<ade::iTensor*,size_t> graphpair)
		{
			return graphpair.second > 0;
		});
	std::list<ade::iFunctor*> parents;
	std::transform(funcsize.begin(), funcsize.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,size_t> graphpair)
		{
			return static_cast<ade::iFunctor*>(graphpair.first);
		});
	parents.sort(
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	std::unordered_map<ade::iTensor*,ade::TensptrT> opt_graph;
	for (ade::iFunctor* func : parents)
	{
		bool changed = false;
		ade::ArgsT children = func->get_children();
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			auto& child = children[i];
			auto tens = child.get_tensor();
			auto it = opt_graph.find(tens.get());
			if (opt_graph.end() != it)
			{
				changed = true;
				children[i] = ade::MappedTensor(
					it->second, child.get_shaper(),
					child.map_io(), child.get_coorder());
			}
		}
		auto opcode = func->get_opcode();
		auto optimized = edit(opcode, children);
		// only record optimization if changed
		if (nullptr != optimized)
		{
			opt_graph.emplace(func, optimized);
		}
		else if (changed)
		{
			opt_graph.emplace(func, ade::TensptrT(
				ade::Functor::get(opcode, children)));
		}
	}

	auto it = opt_graph.find(root.get());
	if (opt_graph.end() == it)
	{
		return root;
	}
	return it->second;
}

}

#endif