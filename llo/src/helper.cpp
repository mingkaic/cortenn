#include "llo/generated/api.hpp"
#include "llo/generated/codes.hpp"
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

ade::Tensorptr matmul (ade::Tensorptr a, ade::Tensorptr b)
{
	const ade::Shape& ashape = a->shape();
	const ade::Shape& bshape = b->shape();

	size_t M = bshape.at(0);
	size_t N = ashape.at(1);
	size_t C = ashape.at(0);
	if (C != bshape.at(1))
	{
		err::fatalf("cannot matmul shapes of incompatible common dimension "
			"%s and %s", ashape.to_string().c_str(),
			bshape.to_string().c_str());
	}
	if (C * N != ashape.n_elems())
	{
		err::fatalf("cannot matmul ashape %s of dimension "
			"higher than 2-D", ashape.to_string().c_str());
	}
	if (C * M != bshape.n_elems())
	{
		err::fatalf("cannot matmul bshape %s of dimension "
			"higher than 2-D", bshape.to_string().c_str());
	}

	return age::reduce_sum(
		age::mul(
			age::permute(age::extend(a, 2, {bshape.at(0)}), {2,1,0}),
			age::permute(age::extend(b, 2, {ashape.at(1)}), {0,2,1})
		), 2);
}

ade::Tensorptr convolve (ade::Tensorptr img, ade::Tensorptr kernel)
{
	const ade::Shape& imgshape = img->shape();
	const ade::Shape& kernelshape = kernel->shape();

	size_t M = imgshape.at(0);
	size_t N = imgshape.at(1);
	size_t m = kernelshape.at(0);
	size_t n = kernelshape.at(1);
	if (M * N != imgshape.n_elems())
	{
		err::fatalf("cannot convolve image shape %s of dimension "
			"higher than 2-D", imgshape.to_string().c_str());
	}
	if (m * n != kernelshape.n_elems())
	{
		err::fatalf("cannot convolve kernel shape %s of dimension "
			"higher than 2-D", imgshape.to_string().c_str());
	}
	if (M < m || N < m)
	{
		err::fatalf("cannot convolve kernel %s against a smaller image %s "
			"in first 2-D", imgshape.to_string().c_str(),
			kernelshape.to_string().c_str());
	}

	uint8_t resm = M - m;
	uint8_t resn = N - n;
	ade::CoordPtrT img_mapper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			fwd[0][0] = fwd[1][1] = fwd[2][0] = fwd[3][1] = 1;
			fwd[2][2] = (double) 1 / resm;
			fwd[3][3] = (double) 1 / resn;
			for (size_t i = 4; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
		}));

	ade::Tensorptr res = ade::Functor::get(ade::Opcode{"PROD", age::PROD},
		{{img_mapper, img}, {ade::extend(2, {resm, resn}), kernel}});

	return age::reduce_sum(age::permute(res, {2, 3, 0, 1}), 2);
}

}

#endif
