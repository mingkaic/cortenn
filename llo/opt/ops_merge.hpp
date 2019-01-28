#include "opt/graph_edit.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_OPS_MERGE_HPP
#define LLO_OPS_MERGE_HPP

namespace llo
{

ade::TensptrT ops_merge_edit (ade::Opcode opcode, ade::ArgsT args);

ade::TensptrT ops_merge (ade::TensptrT root);

}

#endif // LLO_OPS_MERGE_HPP
