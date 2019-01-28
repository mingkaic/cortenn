///
///	zero_prune.hpp
///	llo
///
///	Purpose:
///	Define llo zero pruning functions
///

#include "opt/graph_edit.hpp"

#include "llo/variable.hpp"

#ifndef LLO_ZERO_PRUNE_HPP
#define LLO_ZERO_PRUNE_HPP

namespace llo
{

ade::TensptrT zero_prune_edit (ade::Opcode opcode, ade::ArgsT args);

/// Return tree that prunes zero branches in input according to OPCODE
/// For example, add(x, 0) is converted to simply x, while mul(x, 0) is 0
ade::TensptrT zero_prune (ade::TensptrT root);

}

#endif // LLO_ZERO_PRUNE_HPP
