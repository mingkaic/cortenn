#include <set>

#include "opt/graph_edit.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_OPS_MERGE_HPP
#define LLO_OPS_MERGE_HPP

namespace llo
{

static const std::set<age::_GENERATED_OPCODE> nnary_codes = {
	age::SUM,
	age::PROD,
	age::MIN,
	age::MAX,
};

ade::TensptrT ops_merge_edit (bool& is_optimized,
	ade::Opcode& opcode, ade::ArgsT& args);

ade::TensT ops_merge (ade::TensT roots);

}

#endif // LLO_OPS_MERGE_HPP
