///
/// data.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>

#include "ade/ade.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_COMMON_HPP
#define PBM_COMMON_HPP

namespace pbm
{

/// Tensptr vector type
using TensT = std::vector<ade::TensptrT>;

/// Data serialization functor
using DataSaverT = std::function<std::string(const char*,size_t,size_t)>;

/// Data deserialization functor
using DataLoaderT = std::function<ade::TensptrT(const char*,ade::Shape,\
	size_t,std::string)>;

/// String list type used for paths
using StringsT = std::list<std::string>;

}

#endif // PBM_COMMON_HPP
