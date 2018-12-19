///
///	zprune.hpp
///	llo
///
///	Purpose:
///	Define llo zero pruning functions
///

#include "opt/shear.hpp"

#include "llo/generated/grader.hpp"

#include "llo/data.hpp"

#ifndef LLO_ZPRUNE_HPP
#define LLO_ZPRUNE_HPP

namespace llo
{

/// Return tree that prunes zero branches in input according to OPCODE
/// For example, add(x, 0) is converted to simply x, while mul(x, 0) is 0
ade::TensptrT zero_prune (ade::TensptrT root);

ade::TensptrT derive (ade::TensptrT root, ade::TensptrT target);

ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target);

}

#endif // LLO_ZPRUNE_HPP
