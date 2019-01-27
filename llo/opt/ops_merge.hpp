#include <stack>

#include "opt/graph_edit.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_OPMERGE_HPP
#define LLO_OPMERGE_HPP

namespace llo
{

ade::TensptrT ops_merge_edit (ade::Opcode opcode, ade::ArgsT args);

ade::TensptrT ops_merge (ade::TensptrT root);

}

#endif // LLO_OPMERGE_HPP
