#include "opt/graph_edit.hpp"

#include "llo/opt/const_merge.hpp"
#include "llo/opt/ops_merge.hpp"
#include "llo/opt/one_prune.hpp"
#include "llo/opt/zero_prune.hpp"

#include "llo/variable.hpp"

#ifndef LLO_MULTI_OPT_HPP
#define LLO_MULTI_OPT_HPP

namespace llo
{

ade::TensT multi_optimize (ade::TensT roots,
	std::vector<opt::EditFuncT> edits = {
		const_merge_edit,
		zero_prune_edit,
		one_prune_edit,
		ops_merge_edit,
	});

}

#endif // LLO_MULTI_OPT_HPP
