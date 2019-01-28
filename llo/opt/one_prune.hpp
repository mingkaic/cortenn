///
///	one_prune.hpp
///	llo
///
///	Purpose:
///	Define llo one pruning functions
///

#include "opt/graph_edit.hpp"

#include "llo/variable.hpp"

#ifndef LLO_ONE_PRUNE_HPP
#define LLO_ONE_PRUNE_HPP

namespace llo
{

ade::TensptrT one_prune_edit (ade::Opcode opcode, ade::ArgsT args);

/// Return tree that prunes one branches in input according to OPCODE
/// For example, mul(x, 1) is converted to simply x, while abs(1) is 1
ade::TensptrT one_prune (ade::TensptrT root);

}

#endif // LLO_ONE_PRUNE_HPP
