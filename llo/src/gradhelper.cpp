#include "llo/generated/api.hpp"
#include "llo/generated/codes.hpp"

#include "llo/helper.hpp"
#include "llo/gradhelper.hpp"

#ifdef LLO_GRADHELPER_HPP

namespace llo
{

ade::TensptrT grad_fast_matmul (ade::MappedTensor bwd, ade::TensT args, size_t idx)
{
	auto a = args[0];
	auto b = args[1];
	auto ext_a = age::permute(age::extend(a, 2, {b->shape().at(0)}), {2,1,0});
	auto ext_b = age::permute(age::extend(b, 2, {a->shape().at(1)}), {0,2,1});

	auto ext_fwd = age::mul(
		ext_a,
		ext_b
	);

	auto ext_bwd = age::extend(ade::TensptrT(ade::Functor::get(
		ade::Opcode{"SUM", age::SUM}, {bwd})), 2, {a->shape().at(0)});

	ade::TensptrT ext;
	std::vector<uint8_t> perm;
	if (0 == idx)
	{
		ext = ext_a;
		perm = {2, 1, 0};
	}
	else
	{
		ext = ext_b;
		perm = {0, 2, 1};
	}

	return age::reduce_sum(
		age::permute(
			age::mul(
				age::div(ext_fwd, ext),
				ext_bwd
			), perm), 2);
}

}

#endif
