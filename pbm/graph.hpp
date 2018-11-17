///
/// graph.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal equation graph
///

#include "llo/data.hpp"

#include "bwd/grader.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_GRAPH_HPP
#define PBM_GRAPH_HPP

/// Marshal all equation graphs in roots vector to protobuf object
void save_graph (tenncor::Graph& out, age::TensT& roots);

/// Return all nodes in graph unmarshalled from protobuf object
age::TensT load_graph (const tenncor::Graph& in);

#endif // PBM_GRAPH_HPP
