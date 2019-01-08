///
/// shear.hpp
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

/// Functor for finding leaf targets
using IsLeafTargetT = std::function<bool(ade::iLeaf*)>;

/// Type for mapping function nodes in path to boolean vector
using ParentMapT =

/// Find leaf nodes by some attribute associated to leaf
struct LeafFinder final : public ade::iTraveler
{
	LeafFinder (IsLeafTargetT check_leaf) :
		check_leaf_(check_leaf) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (check_leaf_(leaf))
		{
			founds_.emplace(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (parents_.end() == parents_.find(func))
		{
			auto& children = func->get_children();
			size_t n = children.size();
			bool onpath = false;
			for (size_t i = 0; i < n; ++i)
			{
				ade::TensptrT tens = children[i].get_tensor();
				tens->accept(*this);
				onpath = onpath ||
					(parents_.end() != parents_.find(tens.get()) ||
					founds_.end() != founds_.find(tens.get()));
			}
			if (onpath)
			{
				parents_.emplace(func);
			}
		}
	}

	/// Leaf value getter
	IsLeafTargetT check_leaf_;

	/// Set of leaf nodes found
	std::unordered_set<ade::iTensor*> founds_;

	/// Map of parent nodes in path
	std::unordered_set<ade::iTensor*> parents_;
};

/// Edit functor type
using EditFuncT = std::function<ade::TensptrT(ade::iFunctor*,ade::ArgsT)>;

/// For some target extractable from iLeaf, prune graph such that reduces the
/// length of branches to target from root
/// For example, prune zeros branches by reducing f(x) * 0 to 0,
/// repeat for every instance of multiplication by zero in graph
struct TargetedEdit
{
	TargetedEdit (IsLeafTargetT check_target, EditFuncT edit) :
		finder_(check_target), edit_(edit) {}

	/// Prune graph of root Tensptr
	ade::TensptrT prune (ade::TensptrT root)
	{
		// assert that context will be unaffected by prune,
		// since source will never be touched
		root->accept(finder_);
		auto& pathset = finder_.parents_;
		if (pathset.empty()) // not path to target or root is not a parent
		{
			return root;
		}
		ade::GraphStat stat;
		root->accept(stat);
		// grab the intersection of stat.funcs_ and pathmap
		std::list<ade::iFunctor*> parents;
		std::transform(pathset.begin(), pathset.end(),
			std::back_inserter(parents),
			[](ade::iTensor* tens)
			{
				return static_cast<ade::iFunctor*>(tens);
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
	/// Target finding traveler
	LeafFinder finder_;

	/// Edit functor defining how to edit a function given its new arguments
	EditFuncT edit_;
};

}

#endif // OPT_SHEAR_HPP
