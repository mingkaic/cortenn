#include "llo/opt/zero_prune.hpp"
#include "llo/opt/one_prune.hpp"
#include "llo/opt/ops_merge.hpp"

#include "llo/generated/grader.hpp"

#ifndef LLO_DERIVE_HPP
#define LLO_DERIVE_HPP

namespace llo
{

/// Derive root with respect to target and optimized
ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target);

}

#endif // LLO_DERIVE_HPP
