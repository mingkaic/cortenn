#include <unordered_set>
#include <list>
#include <queue>
#include <chrono>

#include "err/log.hpp"

#include "llo/generated/api.hpp"
#include "llo/operator.hpp"

#include "pbm/graph.hpp"
#include "pbm/source.hpp"

#ifdef PBM_GRAPH_HPP

static std::string make_uid (void* ptr, llo::EngineT& engine)
{
	static std::uniform_int_distribution<short> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) ptr;

	for (size_t i = 0; i < 16; i++)
	{
		short token = tok_dist(engine);
		ss << std::hex << token;
	}
	return ss.str();
}

void save_coord (google::protobuf::RepeatedField<double>* coord,
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

ade::CoordPtrT load_coord (
	const google::protobuf::RepeatedField<double>& coord)
{
	if (ade::mat_dim * ade::mat_dim != coord.size())
	{
		err::fatal("cannot deserialize non-matrix coordinate map");
	}
	return std::make_shared<ade::CoordMap>(
		[&](ade::MatrixT fwd)
		{
			for (uint8_t i = 0; i < ade::mat_dim; ++i)
			{
				for (uint8_t j = 0; j < ade::mat_dim; ++j)
				{
					fwd[i][j] = coord[i * ade::mat_dim + j];
				}
			}
		});
}

struct GraphDFSOrder final : public ade::iTraveler
{
	/// Implemenation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
		if (visited_.end() == visited_.find(leaf))
		{
			leaves_.push_back(leaf);
			visited_.emplace(leaf);
		}
	}

	/// Implemenation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (visited_.end() == visited_.find(func))
		{
			funcs_.push_back(func);
			visited_.emplace(func);

			ade::ArgsT children = func->get_children();
			for (auto& child : children)
			{
				child.tensor_->accept(*this);
			}
		}
	}

	// List of leaves visited (left to right)
	std::list<ade::Tensor*> leaves_;

	// List of functions visited (by depth-first)
	std::list<ade::iFunctor*> funcs_;

	// Visited nodes
	std::unordered_set<ade::iTensor*> visited_;
};

void save_graph (tenncor::Graph& out, age::TensT& roots)
{
	ade::GraphStat stat;
	GraphDFSOrder order;

	for (ade::Tensorptr& tens : roots)
	{
		tens->accept(stat);
		tens->accept(order);
	}

	// sort functions from the root with the smallest subtree to the largest
	// this ensures every children of a node appears before the parent,
	// as is the order of node creations
	order.funcs_.sort(
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	std::vector<ade::iFunctor*> funcs(
		order.funcs_.begin(), order.funcs_.end());
	std::vector<ade::Tensor*> leaves(
		order.leaves_.begin(), order.leaves_.end());

	// all nodes in leaf appear before funcs
	std::unordered_map<ade::iTensor*,size_t> ordermap;
	size_t nleaves = leaves.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		ade::Tensor* tens = leaves[i];
		ordermap[tens] = i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Source* src = pb_node->mutable_source();
		save_data(src, tens);
	}
	for (size_t i = 0, n = funcs.size(); i < n; ++i)
	{
		ade::iFunctor* f = funcs[i];
		ordermap[f] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Functor* func = pb_node->mutable_functor();
		func->set_opname(f->get_opcode().name_);
		const ade::ArgsT& children = f->get_children();
		for (auto& child : children)
		{
			tenncor::NodeArg* arg = func->add_args();
			ade::iTensor* tens = child.tensor_.get();
			arg->set_idx(ordermap[tens]);
			save_coord(arg->mutable_coord(), child.mapper_);
		}
	}
	out.set_id(make_uid(&out, llo::get_engine()));
}

age::TensT load_graph (const tenncor::Graph& in)
{
	auto nodes = in.nodes();
	age::TensT outvec;
	for (const tenncor::Node& node : nodes)
	{
		if (node.has_source())
		{
			const tenncor::Source& source = node.source();
			outvec.push_back(load_source(source));
		}
		else
		{
			tenncor::Functor func = node.functor();
			auto nodeargs = func.args();
			ade::ArgsT args;
			for (auto nodearg : nodeargs)
			{
				ade::CoordPtrT coord = load_coord(nodearg.coord());
				args.push_back({coord, outvec[nodearg.idx()]});
			}
			outvec.push_back(ade::Functor::get(ade::Opcode{func.opname(),
				age::get_op(func.opname())}, args));
		}
	}
	return outvec;
}

#endif
