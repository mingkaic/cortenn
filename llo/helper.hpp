#include "ade/ifunctor.hpp"

#ifndef LLO_HELPER_HPP
#define LLO_HELPER_HPP

namespace llo
{

ade::TensptrT grad_prod (size_t gradidx, ade::TensT tens);

ade::TensptrT grad_min (size_t gradidx, ade::TensT tens);

ade::TensptrT grad_max (size_t gradidx, ade::TensT tens);

ade::CoordPtrT reduce (uint8_t rank, const ade::Shape& shape);

ade::TensptrT matmul (ade::TensptrT a, ade::TensptrT b);

ade::TensptrT convolve (ade::TensptrT img, ade::TensptrT kernel);

}

#endif // LLO_HELPER_HPP
