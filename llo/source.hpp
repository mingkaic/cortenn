///
/// source.hpp
/// llo
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "ade/ileaf.hpp"

#include "llo/data.hpp"

#ifndef LLO_SOURCE_HPP
#define LLO_SOURCE_HPP

namespace llo
{

/// Marshal iSource to tenncor::Source
std::string serialize (const char* in, size_t nelems, size_t typecode);

/// Unmarshal tenncor::Source as Variable containing context of source
ade::TensptrT deserialize (const char* pb, ade::Shape shape,
    size_t typecode, std::string label);

}

#endif // LLO_SOURCE_HPP
