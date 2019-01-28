#include "llo/opt/const_merge.hpp"

#include "llo/constant.hpp"
#include "llo/eval.hpp"

#ifdef LLO_CONST_MERGE_HPP

namespace llo
{

ade::TensptrT const_merge_edit (ade::Opcode opcode, ade::ArgsT args)
{
	if (std::all_of(args.begin(), args.end(),
		[](ade::MappedTensor& arg)
		{
			return nullptr != dynamic_cast<llo::Constant*>(
				arg.get_tensor().get());
		}))
	{
		ade::TensptrT temp(ade::Functor::get(opcode, args));
		auto tens = llo::eval<double>(temp);
		return ade::TensptrT(Constant::get(
			(char*) tens->data(), age::DOUBLE, temp->shape()));
	}
	return nullptr;
}

ade::TensptrT const_merge (ade::TensptrT root)
{
	return opt::graph_edit(root, const_merge_edit);
}

}

#endif
