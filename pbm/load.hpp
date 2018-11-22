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

using LoadTensT = std::pair<ade::TensptrT,StringsT>;

using LoadVecsT = std::vector<LoadTensT>;

/// Return all nodes in graph unmarshalled from protobuf object
LoadVecsT load_graph (const tenncor::Graph& in, iDataLoader& dataloader);

}

#endif // PBM_GRAPH_HPP
