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

using LabelTensT = std::unordered_map<std::string,ade::Tensorptr>;

/// Return all nodes in graph unmarshalled from protobuf object
LabelTensT load_graph (const tenncor::Graph& in, DataLoaderPtrT dataloader);

}

#endif // PBM_GRAPH_HPP
