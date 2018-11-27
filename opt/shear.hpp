///
/// shear.hpp
/// opt
///
/// Purpose:
/// Define ade graph pruning functions
///

#include "ade/ade.hpp"

#ifndef OPT_SHEAR_HPP
#define OPT_SHEAR_HPP

namespace opt
{

/// Functor for getting leaf values
template <typename T>
using GetLeafValT = std::function<T(ade::iLeaf*)>;

/// Type for mapping function nodes in path to boolean vector
using ParentMapT = std::unordered_map<ade::iTensor*,std::unordered_set<size_t>>;

using PruneFuncT = std::function<ade::TensptrT(bool&,ade::iFunctor*,\
	std::unordered_set<size_t>,ade::ArgsT)>;

/// Find leaf nodes by some attribute associated to leaf
template <typename T>
struct LeafFinder final : public ade::iTraveler
{
	LeafFinder (T target, GetLeafValT<T> get_leaf) :
		target_(target), get_leaf_(get_leaf) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (target_ == get_leaf_(leaf))
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
			std::unordered_set<size_t> path;
			for (size_t i = 0; i < n; ++i)
			{
				ade::TensptrT tens = children[i].tensor_;
				tens->accept(*this);
				if (parents_.end() != parents_.find(tens.get()) ||
					founds_.end() != founds_.find(tens.get()))
				{
					path.emplace(i);
				}
			}
			if (false == path.empty())
			{
				parents_[func] = path;
			}
		}
	}

	/// Target of label all paths are travelling to
	T target_;

	/// Leaf value getter
	GetLeafValT<T> get_leaf_;

	/// Set of leaf nodes found
	std::unordered_set<ade::iTensor*> founds_;

	/// Map of parent nodes in path
	ParentMapT parents_;
};

/// For some target extractable from iLeaf, prune graph such that reduces the
/// length of branches to target from root
/// For example, prune zeros branches by reducing f(x) * 0 to 0,
/// repeat for every instance of multiplication by zero in graph
template <typename T>
struct TargetPruner
{
	TargetPruner (T target, GetLeafValT<T> find_target, PruneFuncT pruner) :
		finder_(target, find_target), pruner_(pruner) {}

	ade::TensptrT prune (ade::TensptrT root)
	{
		// assert that context will be unaffected by prune,
		// since source will never be touched
		root->accept(finder_);
		auto& pathmap = finder_.parents_;
		if (pathmap.empty()) // not path to target or root is not a parent
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

		// each proceeding node in parents list is closer to target
		// start pruning according to each parent node in order
		std::unordered_set<ade::iTensor*> targets = finder_.founds_;

		std::unordered_map<ade::iTensor*,ade::TensptrT> mapping;
		std::unordered_map<ade::iTensor*,bool> targetmap;
		for (ade::iFunctor* func : parents)
		{
			ade::ArgsT children = func->get_children();
			std::unordered_set<size_t> indices = pathmap[func]; // indices lead to target nodes
			for (auto it = indices.begin(), et = indices.end(); it != et;)
			{
				ade::MappedTensor& child = children[*it];
				ade::iTensor* tens = child.tensor_.get();
				auto zit = targets.find(tens);
				if (targets.end() == zit) // child is not target, so erase ot from indices
				{
					it = indices.erase(it);
				}
				else
				{
					++it;
				}
			}
			bool is_target = false;
			mapping.emplace(func, pruner_(is_target, func, indices, children));
			if (is_target) // func maps to target, so store in targets
			{
				targets.emplace(func);
			}
		}
		auto it = mapping.find(root.get());
		if (mapping.end() == it)
		{
			logs::fatal("something went wrong"); // todo: probably add context?
		}
		return it->second;
	}

private:
	LeafFinder<T> finder_;

	PruneFuncT pruner_;
};

}

#endif // OPT_SHEAR_HPP
