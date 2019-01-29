#include "llo/opt/const_merge.hpp"
#include "llo/opt/ops_merge.hpp"

#include "llo/constant.hpp"
#include "llo/eval.hpp"

#ifdef LLO_CONST_MERGE_HPP

namespace llo
{

ade::TensptrT const_merge_edit (ade::Opcode opcode, ade::ArgsT args)
{
	ade::ArgsT cargs;
	std::copy_if(args.begin(), args.end(), std::back_inserter(cargs),
		[](ade::MappedTensor& arg)
		{
			return nullptr != dynamic_cast<llo::Constant*>(
				arg.get_tensor().get());
		});
	if (cargs.size() == args.size())
	{
		ade::TensptrT temp(ade::Functor::get(opcode, args));
		auto tens = llo::eval<double>(temp);
		return ade::TensptrT(Constant::get(
			(char*) tens->data(), age::DOUBLE, temp->shape()));
	}
	else if (nnary_codes.find((age::_GENERATED_OPCODE)
		opcode.code_) != nnary_codes.end() && cargs.size() > 2)
	{
		ade::TensptrT temp(ade::Functor::get(opcode, args));
		auto tens = llo::eval<double>(temp);
		ade::TensptrT carg(Constant::get(
			(char*) tens->data(), age::DOUBLE, temp->shape()));

		ade::ArgsT vargs;
		std::copy_if(args.begin(), args.end(), std::back_inserter(vargs),
			[](ade::MappedTensor& arg)
			{
				return nullptr == dynamic_cast<llo::Constant*>(
					arg.get_tensor().get());
			});
		vargs.push_back(ade::identity_map(carg));
		return ade::TensptrT(ade::Functor::get(opcode, vargs));
	}
	return nullptr;
}

ade::TensptrT const_merge (ade::TensptrT root)
{
	return opt::graph_edit(root, const_merge_edit);
}

}

#endif
