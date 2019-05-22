#include "llo/opt/const_merge.hpp"
#include "llo/opt/ops_merge.hpp"

#include "llo/constant.hpp"
#include "llo/eval.hpp"

#ifdef LLO_CONST_MERGE_HPP

namespace llo
{

ade::TensptrT const_merge_edit (bool& is_optimized,
	ade::Opcode& opcode, ade::ArgsT& args)
{
	ade::ArgsT cargs;
	std::copy_if(args.begin(), args.end(), std::back_inserter(cargs),
		[](ade::FuncArg& arg)
		{
			return nullptr != dynamic_cast<Constant*>(
				arg.get_tensor().get());
		});
	if (cargs.size() == args.size())
	{
		ade::TensptrT temp(ade::Functor::get(opcode, args));
		auto tens = eval<double>(temp);
		return ade::TensptrT(Constant::get(
			(char*) tens->data(), age::DOUBLE, temp->shape()));
	}
	else if (nnary_codes.find((age::_GENERATED_OPCODE)
		opcode.code_) != nnary_codes.end() && cargs.size() > 2)
	{
		ade::TensptrT temp(ade::Functor::get(opcode, cargs));
		auto tens = eval<double>(temp);
		ade::TensptrT carg(Constant::get(
			(char*) tens->data(), age::DOUBLE, temp->shape()));

		// assert nnary functions are independent of order
		ade::ArgsT vargs;
		std::copy_if(args.begin(), args.end(), std::back_inserter(vargs),
			[](ade::FuncArg& arg)
			{
				return nullptr == dynamic_cast<Constant*>(
					arg.get_tensor().get());
			});
		vargs.push_back(ade::identity_map(carg));
		is_optimized = true;
		args = vargs;
		return nullptr;
	}
	return nullptr;
}

ade::TensT const_merge (ade::TensT roots)
{
	return opt::graph_edit(roots,
		[](ade::Opcode& opcode,
			ade::ArgsT& args, bool changed) -> ade::TensptrT
		{
			bool is_optimized = false;
			if (auto out = const_merge_edit(is_optimized, opcode, args))
			{
				return out;
			}
			if (changed || is_optimized)
			{
				return ade::TensptrT(ade::Functor::get(opcode, args));
			}
			return nullptr;
		});
}

}

#endif
