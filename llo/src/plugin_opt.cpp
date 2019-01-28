#include "opt/graph_edit.hpp"

#include "llo/opt/const_merge.hpp"
#include "llo/opt/ops_merge.hpp"
#include "llo/opt/one_prune.hpp"
#include "llo/opt/zero_prune.hpp"
#include "llo/opt/plugin_opt.hpp"

#ifdef LLO_PLUGIN_OPT_HPP

namespace llo
{

static const std::vector<opt::EditFuncT> edits =
{
    const_merge_edit,
    zero_prune_edit,
    one_prune_edit,
    ops_merge_edit,
};

ade::TensptrT plugin_edit (ade::Opcode opcode, ade::ArgsT args)
{
    ade::TensptrT out;
    for (auto it = edits.begin(), et = edits.end();
        it != et && nullptr == out; ++it)
    {
        out = (*it)(opcode, args);
    }
    return out;
}

ade::TensptrT plugin_optimize (ade::TensptrT root)
{
	return opt::graph_edit(root, plugin_edit);
}

}

#endif
