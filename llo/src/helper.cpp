#include "llo/generated/api.hpp"
#include "llo/generated/codes.hpp"
#include "llo/helper.hpp"

#ifdef LLO_HELPER_HPP

namespace llo
{

ade::TensptrT grad_prod (size_t gradidx, age::TensT tens)
{
	tens.erase(tens.begin() + gradidx);
	return age::prod(tens);
}

ade::TensptrT grad_min (size_t gradidx, age::TensT tens)
{
	return age::eq(age::min(tens), tens[gradidx]);
}

ade::TensptrT grad_max (size_t gradidx, age::TensT tens)
{
	return age::eq(age::max(tens), tens[gradidx]);
}

ade::CoordPtrT reduce (uint8_t rank, const ade::Shape& shape)
{
	std::vector<ade::DimT> slist(shape.begin() + rank, shape.end());
	return ade::reduce(rank, slist);
}

ade::TensptrT matmul (ade::TensptrT a, ade::TensptrT b)
{
	const ade::Shape& ashape = a->shape();
	const ade::Shape& bshape = b->shape();

	size_t M = bshape.at(0);
	size_t N = ashape.at(1);
	size_t C = ashape.at(0);
	if (C != bshape.at(1))
	{
		logs::fatalf("cannot matmul shapes of incompatible common dimension "
			"%s and %s", ashape.to_string().c_str(),
			bshape.to_string().c_str());
	}
	if (C * N != ashape.n_elems())
	{
		logs::fatalf("cannot matmul ashape %s of dimension "
			"higher than 2-D", ashape.to_string().c_str());
	}
	if (C * M != bshape.n_elems())
	{
		logs::fatalf("cannot matmul bshape %s of dimension "
			"higher than 2-D", bshape.to_string().c_str());
	}

	return age::reduce_sum(
		age::mul(
			age::permute(age::extend(a, 2, {bshape.at(0)}), {2,1,0}),
			age::permute(age::extend(b, 2, {ashape.at(1)}), {0,2,1})
		), 2);
}

// specifications according to https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
// this is to avoid changing rocnnet too much
// (todo: consider simplification after experimenting with rocnnet)
ade::TensptrT convolve (ade::TensptrT img, ade::TensptrT kernel)
{
	const ade::Shape& imgshape = img->shape();
	const ade::Shape& kernelshape = kernel->shape();

	uint8_t nbatch = imgshape.at(0);
	uint8_t in_width = imgshape.at(1);
	uint8_t in_height = imgshape.at(2);
	uint8_t in_channels = imgshape.at(3);

	uint8_t kernel_width = kernelshape.at(0);
	uint8_t kernel_height = kernelshape.at(1);
	uint8_t out_channels = kernelshape.at(3);
	if (in_channels != kernelshape.at(2))
	{
		logs::fatalf("cannot convolve with mismatch img %s and kernel %s "
			"in_channel (dim=3 for img, dim=2 for kernel)",
			imgshape.to_string().c_str(), kernelshape.to_string().c_str());
	}
	if (in_width < kernel_width || in_height < kernel_height)
	{
		logs::fatalf("cannot convolve kernel %s against a smaller image %s",
			kernelshape.to_string().c_str(),
			imgshape.to_string().c_str());
	}

	// map img to shape <nbatch, in_width, in_height, ?(out_channels),
	//		?(kernel_width), ?(kernel_height), in_channels>
	ade::CoordPtrT img_mapper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			fwd[0][0] = fwd[1][1] = fwd[2][2] = fwd[3][6] = 1;
			fwd[4][3] = out_channels;
			fwd[5][4] = kernel_width;
			fwd[6][5] = kernel_height;
			for (uint8_t i = 7; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
		}));

	// map kernel to shape <?(nbatch), ?(in_width), ?(in_height), out_channels,
	//		kernel_width, kernel_height, in_channels>
	ade::CoordPtrT kernel_mapper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			fwd[0][3] = fwd[1][4] = fwd[2][5] = fwd[3][6] = 1;
			fwd[4][0] = nbatch;
			fwd[5][1] = in_width;
			fwd[6][2] = in_height;
			for (uint8_t i = 7; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
		}));

	ade::TensptrT prod(ade::Functor::get(ade::Opcode{"PROD", age::PROD}, {
		{img_mapper, img}, {kernel_mapper, kernel}}));

	return age::reduce_sum(prod, 4);
}

}

#endif
