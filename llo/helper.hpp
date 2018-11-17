#include "ade/ifunctor.hpp"

#ifndef LLO_HELPER_HPP
#define LLO_HELPER_HPP

namespace llo
{

ade::Tensorptr grad_prod (size_t gradidx, age::TensT tens);

ade::Tensorptr grad_min (size_t gradidx, age::TensT tens);

ade::Tensorptr grad_max (size_t gradidx, age::TensT tens);

ade::CoordPtrT reduce (uint8_t rank, const ade::Shape& shape);

}

#endif // LLO_HELPER_HPP
