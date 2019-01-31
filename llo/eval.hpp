///
/// eval.hpp
/// llo
///
/// Purpose:
/// Define evaluation visitor to evaluate the root of an equation graph
///

#include "ade/traveler.hpp"
#include "ade/ifunctor.hpp"

#include "llo/generated/opmap.hpp"

#include "llo/operator.hpp"

#ifndef LLO_EVAL_HPP
#define LLO_EVAL_HPP

namespace llo
{

/// Visitor implementation to evaluate ade nodes according to ctx and dtype
/// Given a global context containing ade-llo association maps, get data from
/// Sources when possible, otherwise treat native ade::iTensors as zeroes
/// Additionally, Evaluator attempts to get meta-data from FuncWrapper
/// before checking native ade::Functor
template <typename T>
struct Evaluator final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		void* ptr = leaf->data();
		age::_GENERATED_DTYPE intype =
			(age::_GENERATED_DTYPE) leaf->type_code();
		const ade::Shape& shape = leaf->shape();
		out_ = raw_to_tensorptr<T>(ptr, intype, shape);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		age::_GENERATED_OPCODE opcode = (age::_GENERATED_OPCODE)
			func->get_opcode().code_;

		const ade::Shape& outshape = func->shape();
		ade::ArgsT children = func->get_children();
		uint8_t nargs = children.size();
		DataArgsT<T> argdata(nargs);
		for (uint8_t i = 0; i < nargs; ++i)
		{
			Evaluator<T> evaler;
			children[i].get_tensor()->accept(evaler);
			argdata[i] = DataArg<T>{
				evaler.out_,
				children[i].get_coorder(),
				children[i].map_io(),
			};
		}

		out_ = get_tensorptr<T>(nullptr, outshape);
		age::typed_exec<T>(opcode, *out_, argdata);
	}

	/// Output data evaluated upon visiting node
	TensptrT<T> out_;
};

/// Evaluate generic data of tens converted to specified type
template <typename T>
TensptrT<T> eval (ade::iTensor* tens)
{
	Evaluator<T> eval;
	tens->accept(eval);
	return eval.out_;
}

/// Evaluate generic data of tens pointer converted to specified type
template <typename T>
TensptrT<T> eval (ade::TensptrT tens)
{
	Evaluator<T> eval;
	tens->accept(eval);
	return eval.out_;
}

}

#endif // LLO_EVAL_HPP
