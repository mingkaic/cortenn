#include <list>
#include <cassert>

#include "ade/functor.hpp"

#include "llo/opt/zero_prune.hpp"

#include "llo/constant.hpp"
#include "llo/eval.hpp"

#ifdef LLO_ZERO_PRUNE_HPP

namespace llo
{

static bool const_is_zero (Constant* cst)
{
	Evaluator<double> eval;
	cst->accept(eval);
	double* ptr = eval.out_->data();
	return std::all_of(ptr, ptr + cst->shape().n_elems(),
		[](double d) { return 0 == d; });
}

// todo: change this to target fixed value instead of looking at label
ade::TensptrT zero_prune_edit (bool& is_optimized,
	ade::Opcode& opcode, ade::ArgsT& args)
{
	size_t n = args.size();
	bool has_zero = false;
	std::vector<bool> is_zero(n, false);
	for (size_t i = 0; i < n; ++i)
	{
		auto cst = dynamic_cast<Constant*>(args[i].get_tensor().get());
		is_zero[i] = nullptr != cst && const_is_zero(cst);
		has_zero = has_zero || is_zero[i];
	}
	if (has_zero)
	{
		switch (opcode.code_)
		{
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::TAN:
			case age::SQRT:
			case age::ROUND:
			case age::PROD:
				return ade::TensptrT(Constant::get(0, args[0].shape()));
			case age::COS:
			case age::EXP:
				return ade::TensptrT(Constant::get(1, args[0].shape()));
			case age::LOG:
				logs::fatal("cannot LOG by zero");
			case age::POW:
				if (is_zero[0])
				{
					return ade::TensptrT(Constant::get(0, args[0].shape()));
				}
				// else if is_zero[1]
				return ade::TensptrT(Constant::get(1, args[1].shape()));
			case age::SUM:
			{
				ade::ArgsT filtered;
				for (size_t i = 0, n = args.size(); i < n; ++i)
				{
					if (false == is_zero[i])
					{
						filtered.push_back(args[i]);
					}
				}
				if (filtered.empty())
				{
					return ade::TensptrT(Constant::get(0, args[0].shape()));
				}
				is_optimized = true;
				opcode = ade::Opcode{"SUM", age::SUM};
				args = filtered;
				return nullptr;
			}
			case age::SUB:
				if (is_zero[0] && is_zero[1])
				{
					return ade::TensptrT(Constant::get(0, args[0].shape()));
				}
				else if (is_zero[0])
				{
					is_optimized = true;
					opcode = ade::Opcode{"NEG", age::NEG};
					args = {args[1]};
					return nullptr;
				}
				// else if is_zero[1]
				return args[0].get_tensor();
			case age::DIV:
				if (is_zero[1])
				{
					logs::fatal("cannot DIV by zero");
				}
				// else if is_zero[0]
				return ade::TensptrT(Constant::get(0, args[0].shape()));
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_UNIF:
			case age::RAND_NORM:
			case age::MATMUL:
				break;
			default:
				logs::fatalf("cannot zero prune unknown opcode \"%s\"",
					opcode.name_.c_str());
		}
	}
	return nullptr;
}

ade::TensptrT zero_prune (ade::TensptrT root)
{
	return opt::graph_edit(root, zero_prune_edit);
}

}

#endif
