#include "llo/opt/multi_opt.hpp"

#ifdef LLO_MULTI_OPT_HPP

namespace llo
{

ade::TensT multi_optimize (ade::TensT roots,
	std::vector<opt::EditFuncT> edits)
{
	return opt::graph_edit(roots,
		[&edits](bool& is_optimized,
			ade::Opcode& opcode, ade::ArgsT& args) -> ade::TensptrT
		{
			for (auto edit : edits)
			{
				if (auto out = edit(is_optimized, opcode, args))
				{
					return out;
				}
			}
			return nullptr;
		});
}

}

#endif
