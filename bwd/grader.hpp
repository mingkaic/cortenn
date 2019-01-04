///
/// grader.hpp
/// bwd
///
/// Purpose:
/// Define grader traveler to build partial derivative equations
///

#include <list>

#include "ade/ade.hpp"

#ifndef BWD_GRADER_HPP
#define BWD_GRADER_HPP

namespace age
{

/// Ruleset used by a Grader traveler to derive equations
struct iRuleSet
{
	virtual ~iRuleSet (void) = default;

	/// Return tensor leaf containing scalar of specific shape
	virtual ade::LeafptrT data (double scalar, ade::Shape shape) = 0;

	/// Return opcode representing nnary sum
	virtual ade::Opcode sum_opcode (void) = 0;

	/// Return d(fwd)/d(x) given:
	/// bwd = d(args[idx])/d(x)
	/// Generally,
	/// d(fwd)/d(x) = rule(fwd,args,idx) * reduction_consolidation(bwd)
	virtual ade::TensptrT chain_rule (ade::iFunctor* fwd,
		ade::MappedTensor bwd, ade::TensT args, size_t idx) = 0;
};

/// Traveler to obtain derivative of accepted node with respect to target
struct Grader final : public ade::iTraveler
{
	Grader (const ade::iTensor* target, std::shared_ptr<iRuleSet> rules) :
		target_(target), rules_(rules)
	{
		if (target_ == nullptr)
		{
			logs::fatal("cannot derive with respect to null");
		}
		if (rules_ == nullptr)
		{
			logs::fatal("cannot derive without ruleset");
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (leaf == target_)
		{
			derivatives_.emplace(leaf,
				rules_->data(1, target_->shape()));
		}
		else
		{
			derivatives_.emplace(leaf,
				rules_->data(0, target_->shape()));
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override;

	/// Target of tensor all visited nodes are derived with respect to
	const ade::iTensor* target_;

	/// Map forward root node to derivative root
	std::unordered_map<const ade::iTensor*,ade::TensptrT> derivatives_;

private:
	/// Ruleset used by this instance
	std::shared_ptr<iRuleSet> rules_;
};

/// Return ArgsT with each tensor in TensT attached to identity mapper
ade::ArgsT to_args (ade::TensT tens);

}

#endif // BWD_GRADER_HPP
