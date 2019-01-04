#include "llo/generated/api.hpp"
#include "llo/generated/codes.hpp"
#include "llo/data.hpp"
#include "llo/helper.hpp"

#ifdef LLO_HELPER_HPP

namespace llo
{

ade::TensptrT mtens_mul (ade::TensptrT lhs, ade::MappedTensor rhs)
{
    return ade::TensptrT(ade::Functor::get(ade::Opcode{"PROD", age::PROD}, {
        ade::identity_map(lhs), rhs
    }));
}

ade::TensptrT grad_prod (ade::iFunctor* fwd, size_t gradidx, ade::TensT tens)
{
	auto fwd_children = fwd->get_children();
	ade::TensptrT fwd_cpy(ade::Functor::get(
		fwd->get_opcode(), fwd_children));

	auto& fwd_child = fwd_children[gradidx];
	ade::MappedTensor fwd_mapped(fwd_cpy,
		ade::CoordptrT(fwd_child.get_shaper()->reverse()),
		!fwd_child.map_io(), fwd_child.get_coorder());

	ade::TensptrT fwd_extended(
		ade::Functor::get(ade::Opcode{"SUM", age::SUM}, {fwd_mapped}));

	return age::div(fwd_extended, tens[gradidx]);
}

ade::TensptrT grad_min (ade::iFunctor* fwd, size_t gradidx, ade::TensT tens)
{
	ade::TensptrT fwd_cpy(ade::Functor::get(
		fwd->get_opcode(), fwd->get_children()));
	auto fchildren = fwd->get_children();
	ade::CoordptrT shaper(fchildren[gradidx].get_shaper()->reverse());
	ade::TensptrT rev_fwd(ade::Functor::get(ade::Opcode{"SUM",age::SUM},
		{ade::MappedTensor(fwd_cpy, shaper)}));
	return age::eq(rev_fwd, tens[gradidx]);
}

ade::TensptrT grad_max (ade::iFunctor* fwd, size_t gradidx, ade::TensT tens)
{
	ade::TensptrT fwd_cpy(ade::Functor::get(
		fwd->get_opcode(), fwd->get_children()));
	auto fchildren = fwd->get_children();
	ade::CoordptrT shaper(fchildren[gradidx].get_shaper()->reverse());
	ade::TensptrT rev_fwd(ade::Functor::get(ade::Opcode{"SUM",age::SUM},
		{ade::MappedTensor(fwd_cpy, shaper)}));
	return age::eq(rev_fwd, tens[gradidx]);
}

ade::TensptrT reduce (ade::Opcode opcode, ade::TensptrT tens, uint8_t dim)
{
	ade::Shape shape = tens->shape();
	std::vector<ade::DimT> slist(shape.begin() + dim, shape.end());
	auto out = ade::Functor::get(opcode, {
		ade::reduce_map(tens, dim, slist),
	});
	return ade::TensptrT(out);
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
ade::TensptrT convolution (ade::TensptrT img, ade::TensptrT kernel)
{
	const ade::Shape& imgshape = img->shape();
	const ade::Shape& kernelshape = kernel->shape();

	uint8_t nbatch = imgshape.at(3);
	uint8_t in_width = imgshape.at(2);
	uint8_t in_height = imgshape.at(1);
	uint8_t in_channels = imgshape.at(0);

	uint8_t kernel_width = kernelshape.at(3);
	uint8_t kernel_height = kernelshape.at(2);
	uint8_t out_channels = kernelshape.at(0);
	if (in_channels != kernelshape.at(1))
	{
		logs::fatalf("cannot convolution with mismatch img %s and kernel %s "
			"in_channel (dim=0 for img, dim=1 for kernel)",
			imgshape.to_string().c_str(), kernelshape.to_string().c_str());
	}
	if (in_width < kernel_width || in_height < kernel_height)
	{
		logs::fatalf("cannot convolution kernel %s against a smaller image %s",
			kernelshape.to_string().c_str(),
			imgshape.to_string().c_str());
	}

	uint8_t invalid_height = 2 * std::floor(kernel_height / 2);
	uint8_t invalid_width = 2 * std::floor(kernel_width / 2);

	// input: [in_channels, in_height, in_width, nbatch]
	// output: [out_channels,
	//			in_height-invalid_height,
	//			in_width-invalid_height,
	//			nbatch, in_channels, kernel_height, kernel_width]
	ade::CoordptrT img_shaper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			fwd[0][4] =
			fwd[1][1] =
			fwd[2][2] =
			fwd[3][3] = 1;
			fwd[4][0] = out_channels;
			fwd[5][5] = kernel_height;
			fwd[6][6] = kernel_width;
			fwd[ade::mat_dim - 1][1] = -invalid_height;
			fwd[ade::mat_dim - 1][2] = -invalid_width;
			for (uint8_t i = 7; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
		}));

	ade::CoordptrT img_mapper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			fwd[1][1] =
			fwd[2][2] =
			fwd[3][3] =
			fwd[4][0] =
			fwd[5][1] =
			fwd[6][2] = 1;
			fwd[0][4] = 1.0 / out_channels;
			fwd[1][5] = 1.0 / (in_height - invalid_height);
			fwd[2][6] = 1.0 / (in_width - invalid_width);
			for (uint8_t i = 7; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
		}));

	// input: [out_channels, in_channels, kernel_height, kernel_width]
	// output: [out_channels,
	//			in_height - invalid_height,
	//			in_width - invalid_width,
	//			nbatch, in_channels, kernel_height, kernel_width]
	ade::CoordptrT kernel_mapper(new ade::CoordMap(
		[&](ade::MatrixT fwd)
		{
			fwd[0][0] =
			fwd[1][4] =
			fwd[2][5] =
			fwd[3][6] =
			fwd[4][1] =
			fwd[5][2] =
			fwd[6][3] = 1;
			fwd[ade::mat_dim - 1][1] = in_height - invalid_height - 1;
			fwd[ade::mat_dim - 1][2] = in_width - invalid_width - 1;
			fwd[ade::mat_dim - 1][3] = nbatch - 1;
			for (uint8_t i = 7; i < ade::mat_dim; ++i)
			{
				fwd[i][i] = 1;
			}
		}));

	ade::TensptrT prod(ade::Functor::get(ade::Opcode{"PROD", age::PROD}, {
		ade::MappedTensor(img, img_shaper, false, img_mapper),
		ade::MappedTensor(kernel, kernel_mapper),
	}));

	return age::reduce_sum(prod, 4);
}

}

#endif
