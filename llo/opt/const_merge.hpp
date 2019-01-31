#include "opt/graph_edit.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_CONST_MERGE_HPP
#define LLO_CONST_MERGE_HPP

namespace llo
{

ade::TensptrT const_merge_edit (bool& is_optimized,
	ade::Opcode& opcode, ade::ArgsT& args);

ade::TensT const_merge (ade::TensT roots);

}

#endif // LLO_CONST_MERGE_HPP
