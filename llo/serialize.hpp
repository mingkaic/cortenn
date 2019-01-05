///
/// serialize.hpp
/// llo
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "llo/data.hpp"

#ifndef LLO_SERIALIZE_HPP
#define LLO_SERIALIZE_HPP

namespace llo
{

/// Marshal data to cortenn::Source
std::string serialize (const char* in, size_t nelems, size_t typecode);

/// Unmarshal cortenn::Source as Variable containing context of source
ade::TensptrT deserialize (const char* pb, ade::Shape shape,
	size_t typecode, std::string label);

}

#endif // LLO_SERIALIZE_HPP
