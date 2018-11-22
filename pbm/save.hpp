///
/// save.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>
#include <unordered_set>

#include "pbm/data.hpp"

#ifndef PBM_SAVE_HPP
#define PBM_SAVE_HPP

namespace pbm
{

using TensLabelT = std::unordered_map<ade::iTensor*,StringsT>;

struct GraphSaver final : public ade::iTraveler
{
	GraphSaver (iDataSaver* saver) :
		saver_(saver) {}

    /// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (visited_.end() == visited_.find(leaf))
		{
            leaf->accept(stat);
			leaves_.push_back(leaf);
			visited_.emplace(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (visited_.end() == visited_.find(func))
		{
            func->accept(stat);
			funcs_.push_back(func);
			visited_.emplace(func);

			ade::ArgsT children = func->get_children();
			for (auto& child : children)
			{
				child.tensor_->accept(*this);
			}
		}
	}

    /// Marshal all equation graphs in roots vector to protobuf object
    void save (tenncor::Graph& out, TensLabelT labels = TensLabelT());

	// List of leaves visited (left to right)
	std::list<ade::iLeaf*> leaves_;

	// List of functions visited (by depth-first)
	std::list<ade::iFunctor*> funcs_;

	// Visited nodes
	std::unordered_set<ade::iTensor*> visited_;

	ade::GraphStat stat;

private:
    void save_coord (
        google::protobuf::RepeatedField<double>* coord,
        const ade::CoordPtrT& mapper);

    /// Marshal llo::iSource to tenncor::Source
    void save_data (tenncor::Node* out, ade::iLeaf* in)
    {
		if (nullptr == saver_)
		{
			err::fatal("cannot save tensor without datasaver");
		}
        saver_->save(*out, in);
    }

    DataSaverPtrT saver_;
};

}

#endif // PBM_SAVE_HPP
