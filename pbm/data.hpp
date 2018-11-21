///
/// save.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include "ade/itensor.hpp"
#include "ade/traveler.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_COMMON_HPP
#define PBM_COMMON_HPP

namespace pbm
{

using TensT = std::vector<ade::Tensorptr>;

struct iDataSaver
{
    virtual ~iDataSaver (void) = default;

    virtual void save (tenncor::Source& out, ade::Tensor* tens) = 0;
};

struct iDataLoader
{
    virtual ~iDataLoader (void) = default;

    virtual ade::Tensorptr load (const tenncor::Source& source) = 0;
};

using DataSaverPtrT = std::unique_ptr<iDataSaver>;

using DataLoaderPtrT = std::unique_ptr<iDataLoader>;

}

#endif // PBM_COMMON_HPP
