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

using PathedMapT = std::unordered_map<ade::TensptrT,StringsT>;

struct GraphSaver final : public ade::iTraveler
{
	GraphSaver (DataSaverT saver) : saver_(saver) {}

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
	void save (tenncor::Graph& out, PathedMapT labels = PathedMapT());

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
	void save_data (tenncor::Source& out, ade::iLeaf* in)
	{
		const ade::Shape& shape = in->shape();
		char* data = (char*) in->data();
		size_t nelems = shape.n_elems();
		size_t tcode = in->type_code();
		out.set_shape(std::string(shape.begin(), shape.end()));
		out.set_data(saver_(data, nelems, tcode));
		out.set_typecode(tcode);
	}

	DataSaverT saver_;
};

}

#endif // PBM_SAVE_HPP
