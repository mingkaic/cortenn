#include "err/log.hpp"

#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#include "pbm/save.hpp"

#ifdef PBM_SAVE_HPP

namespace pbm
{

void GraphSaver::save (tenncor::Graph& out, TensLabelT labels)
{
    // sort functions from the root with the smallest subtree to the largest
    // this ensures every children of a node appears before the parent,
    // as is the order of node creations
    funcs_.sort(
        [&](ade::iTensor* a, ade::iTensor* b)
        {
            return stat.graphsize_[a] < stat.graphsize_[b];
        });

    std::vector<ade::iFunctor*> funcs(funcs_.begin(), funcs_.end());
    std::vector<ade::Tensor*> leaves(leaves_.begin(), leaves_.end());

    // all nodes in leaf appear before funcs
    std::unordered_map<ade::iTensor*,size_t> ordermap;
    size_t nleaves = leaves.size();
    for (size_t i = 0; i < nleaves; ++i)
    {
        ade::Tensor* tens = leaves[i];
        ordermap[tens] = i;

        tenncor::Node* pb_node = out.add_nodes();
        auto it = labels.find(tens);
        if (labels.end() != it)
        {
            pb_node->set_label(it->second);
        }
        tenncor::Source* src = pb_node->mutable_source();
        save_data(src, tens);
    }
    for (size_t i = 0, n = funcs.size(); i < n; ++i)
    {
        ade::iFunctor* f = funcs[i];
        ordermap[f] = nleaves + i;

        tenncor::Node* pb_node = out.add_nodes();
        auto it = labels.find(f);
        if (labels.end() != it)
        {
            pb_node->set_label(it->second);
        }
        tenncor::Functor* func = pb_node->mutable_functor();
        ade::Opcode opcode = f->get_opcode();
        func->set_opname(opcode.name_);
        func->set_opcode(opcode.code_);
        const ade::ArgsT& children = f->get_children();
        for (auto& child : children)
        {
            tenncor::NodeArg* arg = func->add_args();
            ade::iTensor* tens = child.tensor_.get();
            arg->set_idx(ordermap[tens]);
            save_coord(arg->mutable_coord(), child.mapper_);
        }
    }
}

void GraphSaver::save_coord (google::protobuf::RepeatedField<double>* coord,
	const ade::CoordPtrT& mapper)
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
