#include "ade/ifunctor.hpp"

#ifndef LLO_HELPER_HPP
#define LLO_HELPER_HPP

namespace llo
{

/// Return the gradient for prod operation assuming the target derived wrt is
/// index gradidx and arguments are tens
ade::TensptrT grad_prod (size_t gradidx, ade::TensT tens);

/// Return the gradient for min operation assuming the target derived wrt is
/// index gradidx and arguments are tens
ade::TensptrT grad_min (ade::iFunctor* fwd, size_t gradidx, ade::TensT tens);

/// Return the gradient for max operation assuming the target derived wrt is
/// index gradidx and arguments are tens
ade::TensptrT grad_max (ade::iFunctor* fwd, size_t gradidx, ade::TensT tens);

/// Return reduction of tens after dimension dim using opcode operation
ade::TensptrT reduce (ade::Opcode opcode, ade::TensptrT tens, uint8_t dim);

/// Return matmul of a and b
ade::TensptrT matmul (ade::TensptrT a, ade::TensptrT b);

/// Return convolution operation on img with kernel
ade::TensptrT convolution (ade::TensptrT img, ade::TensptrT kernel);

}

#endif // LLO_HELPER_HPP
