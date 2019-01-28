///
/// serialize.hpp
/// llo
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "llo/variable.hpp"

#ifndef LLO_SERIALIZE_HPP
#define LLO_SERIALIZE_HPP

namespace llo
{

/// Marshal data to cortenn::Source
std::string serialize (bool& is_const, ade::iLeaf* leaf);

/// Unmarshal cortenn::Source as Variable containing context of source
ade::TensptrT deserialize (const char* pb, ade::Shape shape,
	size_t typecode, std::string label, bool is_const);

}

#endif // LLO_SERIALIZE_HPP
