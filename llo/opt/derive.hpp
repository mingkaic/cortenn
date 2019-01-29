#include "llo/opt/multi_opt.hpp"

#include "llo/generated/grader.hpp"

#ifndef LLO_DERIVE_HPP
#define LLO_DERIVE_HPP

namespace llo
{

/// Derive root with respect to target and optimized
ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target);

}

#endif // LLO_DERIVE_HPP
