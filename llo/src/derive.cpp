#include "llo/opt/derive.hpp"

#ifdef LLO_DERIVE_HPP

namespace llo
{

ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target)
{
	age::Grader grader(target, std::make_shared<age::RuleSet>());
	root->accept(grader);
	auto it = grader.derivatives_.find(root.get());
	assert(grader.derivatives_.end() != it);
	return ops_reuse(multi_optimize(it->second));
}

}

#endif
