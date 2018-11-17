#include "llo/generated/api.hpp"
#include "llo/helper.hpp"

#ifdef LLO_HELPER_HPP

namespace llo
{

ade::Tensorptr grad_prod (size_t gradidx, age::TensT tens)
{
	tens.erase(tens.begin() + gradidx);
	return age::prod(tens);
}

ade::Tensorptr grad_min (size_t gradidx, age::TensT tens)
{
	return age::eq(age::min(tens), tens[gradidx]);
}

ade::Tensorptr grad_max (size_t gradidx, age::TensT tens)
{
	return age::eq(age::max(tens), tens[gradidx]);
}

ade::CoordPtrT reduce (uint8_t rank, const ade::Shape& shape)
{
	std::vector<ade::DimT> slist(shape.begin() + rank, shape.end());
	return ade::reduce(rank, slist);
}

}

#endif
