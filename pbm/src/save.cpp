#include "logs/logs.hpp"

#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#include "pbm/save.hpp"

#ifdef PBM_SAVE_HPP

namespace pbm
{

void GraphSaver::save (tenncor::Graph& out, PathedMapT labels)
{
	std::unordered_map<ade::iTensor*,StringsT> raw_labels;
	for (auto lpair : labels)
	{
		raw_labels[lpair.first.get()] = lpair.second;
	}

	// sort functions from the root with the smallest subtree to the largest
	// this ensures every children of a node appears before the parent,
	// as is the order of node creations
	funcs_.sort(
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	std::vector<ade::iFunctor*> funcs(funcs_.begin(), funcs_.end());
	std::vector<ade::iLeaf*> leaves(leaves_.begin(), leaves_.end());

	// all nodes in leaf appear before funcs
	std::unordered_map<ade::iTensor*,size_t> ordermap;
	size_t nleaves = leaves.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		ade::iLeaf* tens = leaves[i];
		ordermap[tens] = i;

		tenncor::Node* pb_node = out.add_nodes();
		auto it = raw_labels.find(tens);
		if (raw_labels.end() != it)
		{
			google::protobuf::RepeatedPtrField<std::string> vec(
				it->second.begin(), it->second.end());
			pb_node->mutable_labels()->Swap(&vec);
		}
		save_data(*pb_node->mutable_source(), tens);
	}
	for (size_t i = 0, n = funcs.size(); i < n; ++i)
	{
		ade::iFunctor* f = funcs[i];
		ordermap[f] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		auto it = raw_labels.find(f);
		if (raw_labels.end() != it)
		{
			google::protobuf::RepeatedPtrField<std::string> vec(
				it->second.begin(), it->second.end());
			pb_node->mutable_labels()->Swap(&vec);
		}
		tenncor::Functor* func = pb_node->mutable_functor();
		ade::Opcode opcode = f->get_opcode();
		func->set_opname(opcode.name_);
		func->set_opcode(opcode.code_);
		const ade::ArgsT& children = f->get_children();
		for (auto& child : children)
		{
			tenncor::NodeArg* arg = func->add_args();
			ade::iTensor* tens = child.get_tensor().get();
			arg->set_idx(ordermap[tens]);
			save_coord(arg->mutable_coord(), child.get_coorder());
			if (child.get_shaper() != child.get_coorder())
			{
				save_coord(arg->mutable_shaper(), child.get_shaper());
			}
			arg->set_fwd(child.map_io());
		}
	}
}

void GraphSaver::save_coord (google::protobuf::RepeatedField<double>* coord,
	const ade::CoordptrT& mapper)
{
	mapper->access([coord](const ade::MatrixT& mat)
	{
		for (uint8_t i = 0; i < ade::mat_dim; ++i)
		{
			for (uint8_t j = 0; j < ade::mat_dim; ++j)
			{
				(*coord->Add()) = mat[i][j];
			}
		}
	});
}

}

#endif
