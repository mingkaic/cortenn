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

using DataSaverT = std::function<std::string(const char*,size_t,size_t)>;

using DataLoaderT = std::function<ade::TensptrT(const char*,ade::Shape,\
    size_t,std::string)>;

struct iDataSaver
{
    virtual ~iDataSaver (void) = default;

    virtual std::string serialize (const char* in,
        size_t nelems, size_t typecode) = 0;
};

struct iDataLoader
{
    virtual ~iDataLoader (void) = default;

    virtual ade::TensptrT deserialize (const char* pb,
        ade::Shape shape, size_t typecode, std::string label) = 0;
};

using DataSaverPtrT = std::unique_ptr<iDataSaver>;

using StringsT = std::list<std::string>;

}

#endif // PBM_COMMON_HPP
