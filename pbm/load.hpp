///
/// graph.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal equation graph
///

#include "pbm/data.hpp"

#ifndef PBM_LOAD_HPP
#define PBM_LOAD_HPP

namespace pbm
{

using LabelledTensT = std::pair<ade::TensptrT,StringsT>;

using LabelledsT = std::vector<LabelledTensT>;

struct GraphInfo final
{
	std::unordered_set<ade::TensptrT> roots_;

	LabelledsT labelled_;
};

/// Return all nodes in graph unmarshalled from protobuf object
void load_graph (GraphInfo& out, const tenncor::Graph& in,
	DataLoaderT dataloader);

}

#endif // PBM_GRAPH_HPP
