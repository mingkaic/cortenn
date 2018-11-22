///
/// save.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>

#include "ade/itensor.hpp"
#include "ade/traveler.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_COMMON_HPP
#define PBM_COMMON_HPP

namespace pbm
{

using TensT = std::vector<ade::TensptrT>;

struct iDataSaver
{
    virtual ~iDataSaver (void) = default;

    virtual void save (tenncor::Node& out, ade::iLeaf* tens) = 0;
};

struct iDataLoader
{
    virtual ~iDataLoader (void) = default;

    virtual ade::TensptrT load (const tenncor::Source& source,
        std::string label) = 0;
};

using DataSaverPtrT = std::unique_ptr<iDataSaver>;

using StringsT = std::list<std::string>;

}

#endif // PBM_COMMON_HPP
