#include "bwd/grader.hpp"

#include "llo/opt/derive.hpp"

#ifdef LLO_DERIVE_HPP

namespace llo
{

struct RuleSet final : public age::iRuleSet
{
	ade::LeafptrT data (double scalar, ade::Shape shape) override
	{
		return ade::LeafptrT(llo::Constant::get(scalar,shape));
	}

	ade::Opcode sum_opcode (void) override
	{
		return ade::Opcode{"SUM", age::SUM};
	}

	ade::TensptrT chain_rule (ade::iFunctor* fwd,
		ade::FuncArg bwd, ade::TensT args, size_t idx) override
	{
		return age::chain_rule(fwd, bwd, args, idx);
	}
};

ade::TensptrT derive (ade::TensptrT root, ade::iTensor* target)
{
	age::Grader grader(target, std::make_shared<RuleSet>());
	root->accept(grader);
	auto it = grader.derivatives_.find(root.get());
	assert(grader.derivatives_.end() != it);
	return ops_reuse(multi_optimize({it->second}))[0];
}

ade::TensT multi_derive (ade::TensptrT root, std::vector<ade::iTensor*> targets)
{
	ade::TensT derivatives(targets.size());
	std::transform(targets.begin(), targets.end(), derivatives.begin(),
		[&root](ade::iTensor* target)
		{
			age::Grader grader(target, std::make_shared<RuleSet>());
			root->accept(grader);
			auto it = grader.derivatives_.find(root.get());
			assert(grader.derivatives_.end() != it);
			return it->second;
		});
	return ops_reuse(multi_optimize(derivatives));
}

}

#endif
