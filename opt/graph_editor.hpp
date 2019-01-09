///
/// graph_editor.hpp
/// opt
///
/// Purpose:
/// Define ade graph pruning functions
///

#include <list>

#include "ade/ade.hpp"

#ifndef OPT_SHEAR_HPP
#define OPT_SHEAR_HPP

namespace opt
{

/// Edit functor type
using EditFuncT = std::function<ade::TensptrT(ade::iFunctor*,ade::ArgsT)>;

/// For some target extractable from iLeaf, prune graph such that reduces the
/// length of branches to target from root
/// For example, prune zeros branches by reducing f(x) * 0 to 0,
/// repeat for every instance of multiplication by zero in graph
struct GraphEditor
{
	GraphEditor (EditFuncT edit) : edit_(edit) {}

	/// Prune graph of root Tensptr
	ade::TensptrT edit (ade::TensptrT root)
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

		std::unordered_map<ade::iTensor*,ade::TensptrT> edited_mapping;
		for (ade::iFunctor* func : parents)
		{
			bool func_edited = false; // if edited, mark in mapping
			ade::ArgsT children = func->get_children();
			for (size_t i = 0, n = children.size(); i < n; ++i)
			{
				ade::MappedTensor& child = children[i];
				ade::iTensor* tens = child.get_tensor().get();
				// update children arguments if tens is arguments
				auto edited = edited_mapping.find(tens);
				if (edited_mapping.end() != edited)
				{
					// replace edited child
					child = ade::MappedTensor(
						edited->second,
						child.get_shaper(),
						child.map_io(),
						child.get_coorder()
					);
					func_edited = true;
				}
			}
			ade::TensptrT fwd = edit_(func, children);
			if (nullptr != fwd)
			{
				edited_mapping.emplace(func, fwd);
			}
			else if (func_edited)
			{
				edited_mapping.emplace(func, ade::TensptrT(
					ade::Functor::get(func->get_opcode(), children)));
			}
		}
		auto it = edited_mapping.find(root.get());
		if (edited_mapping.end() == it)
		{
			return root;
		}
		return it->second;
	}

private:
	/// Edit functor defining how to edit a function given its new arguments
	EditFuncT edit_;
};

}

#endif // OPT_SHEAR_HPP
