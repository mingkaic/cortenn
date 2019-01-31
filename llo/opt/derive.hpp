#include "llo/opt/multi_opt.hpp"
#include "llo/opt/ops_reuse.hpp"

#include "llo/generated/grader.hpp"

#ifndef LLO_DERIVE_HPP
#define LLO_DERIVE_HPP

namespace llo
{

/// Derive root with respect to target and optimized
ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target);

ade::TensT multi_derive (ade::TensptrT root, std::vector<ade::iTensor*> targets);

}

#endif // LLO_DERIVE_HPP
