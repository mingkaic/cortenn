///
/// source.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "llo/data.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_SOURCE_HPP
#define PBM_SOURCE_HPP

/// Marshal llo::iSource to tenncor::Source
void save_data (tenncor::Source* out, ade::Tensor* in);

/// Unmarshal tenncor::Source as Variable containing context of source
ade::Tensorptr load_source (const tenncor::Source& source);

#endif // PBM_SOURCE_HPP
