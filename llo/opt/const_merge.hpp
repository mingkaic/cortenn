#include "opt/graph_edit.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_CONST_MERGE_HPP
#define LLO_CONST_MERGE_HPP

namespace llo
{

ade::TensptrT const_merge_edit (ade::Opcode opcode, ade::ArgsT args);

ade::TensptrT const_merge (ade::TensptrT root);

}

#endif // LLO_CONST_MERGE_HPP
