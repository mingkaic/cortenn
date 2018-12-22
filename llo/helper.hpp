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
ade::TensptrT grad_min (size_t gradidx, ade::TensT tens);

/// Return the gradient for max operation assuming the target derived wrt is
/// index gradidx and arguments are tens
ade::TensptrT grad_max (size_t gradidx, ade::TensT tens);

/// Return reduce coordinate mapper for shape down to specified rank
ade::CoordPtrT reduce (uint8_t rank, const ade::Shape& shape);

/// Return matmul of a and b
ade::TensptrT matmul (ade::TensptrT a, ade::TensptrT b);

/// Return img convolved with kernel
ade::TensptrT convolve (ade::TensptrT img, ade::TensptrT kernel);

}

#endif // LLO_HELPER_HPP
