#include "llo/opt/multi_opt.hpp"

#ifdef LLO_MULTI_OPT_HPP

namespace llo
{

ade::TensptrT multi_optimize (ade::TensptrT root,
    std::vector<opt::EditFuncT> edits)
{
	return opt::graph_edit(root,
        [&edits](ade::Opcode opcode, ade::ArgsT args)
        {
            ade::TensptrT out;
            for (auto it = edits.begin(), et = edits.end();
                it != et && nullptr == out; ++it)
            {
                out = (*it)(opcode, args);
            }
            return out;
        });
}

}

#endif
