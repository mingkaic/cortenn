#include "llo/opt/multi_opt.hpp"

#ifdef LLO_MULTI_OPT_HPP

namespace llo
{

ade::TensT multi_optimize (ade::TensT roots,
	std::vector<EditFuncT> edits)
{
	return opt::graph_edit(roots,
		[&edits](ade::Opcode& opcode,
			ade::ArgsT& args, bool changed) -> ade::TensptrT
		{
			bool is_optimized = false;
			for (auto edit : edits)
			{
				if (auto out = edit(is_optimized, opcode, args))
				{
					return out;
				}
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
