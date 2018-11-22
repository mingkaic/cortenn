///
///	shear.hpp
///	llo
///
///	Purpose:
///	Define llo graph pruning functions
///

#include <unordered_map>
#include <unordered_set>

#include "llo/generated/grader.hpp"

#include "llo/data.hpp"

#ifndef LLO_SHEAR_HPP
#define LLO_SHEAR_HPP

namespace llo
{

struct LabelFinder final : public ade::iTraveler
{
	/// Type for mapping function nodes in path to boolean vector
	using ParentMapT = std::unordered_map<ade::iTensor*,std::unordered_set<size_t>>;

	LabelFinder (std::string label) : target_(label) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		auto data = static_cast<Variable*>(leaf);
		if (target_ == data->label_)
		{
			labelled_.emplace(leaf);
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
					labelled_.end() != labelled_.find(tens.get()))
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
	std::string target_;

	std::unordered_set<ade::iTensor*> labelled_;

	/// Map of parent nodes in path
	ParentMapT parents_;
};

/// Return tree that prunes zero branches in input according to OPCODE
/// For example, add(x, 0) is converted to simply x, while mul(x, 0) is 0
ade::TensptrT zero_prune (ade::TensptrT root);

ade::TensptrT derive (ade::TensptrT root, ade::TensptrT target);

}

#endif // LLO_SHEAR_HPP
