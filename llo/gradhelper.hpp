#include "ade/ifunctor.hpp"

#ifndef LLO_GRADHELPER_HPP
#define LLO_GRADHELPER_HPP

namespace llo
{

ade::TensptrT grad_fast_matmul (ade::MappedTensor bwd, ade::TensT args, size_t idx);

}

#endif // LLO_GRADHELPER_HPP
