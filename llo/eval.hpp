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

using FuncptrT = std::shared_ptr<ade::iFunctor>;

struct ShortcutFunctor final : public ade::iFunctor
{
	static ShortcutFunctor* get (age::_GENERATED_OPCODE opcode,
		FuncptrT proxy_root, ade::ArgsT entries)
	{
		ade::GraphStat stat;
		proxy_root->accept(stat);
		for (auto& entry : entries)
		{
			auto tens = entry.get_tensor().get();
			if (stat.graphsize_.end() == stat.graphsize_.find(tens))
			{
				logs::fatalf("expected %s entry to be descendant of %s",
					proxy_root->to_string().c_str(),
					tens->to_string().c_str());
			}
		}
		return new ShortcutFunctor(opcode, proxy_root, entries);
	}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return proxy_root_->shape();
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return proxy_root_->to_string();
	}

	/// Implementation of iFunctor
	ade::Opcode get_opcode (void) const override
	{
		return proxy_root_->get_opcode();
	}

	/// Implementation of iFunctor
	const ade::ArgsT& get_children (void) const override
	{
		return proxy_root_->get_children();
	}

	template <typename T>
	TensorT<T> evaluate (void);

private:
	ShortcutFunctor (age::_GENERATED_OPCODE opcode,
		FuncptrT proxy_root, ade::ArgsT& entries) :
		opcode_(opcode), proxy_root_(proxy_root), entries_(entries) {}

	age::_GENERATED_OPCODE opcode_;

	FuncptrT proxy_root_;

	ade::ArgsT entries_;
};

#define TEMPCONVERT(TYPE)std::vector<T>((TYPE*) ptr, (TYPE*) ptr + n)

/// Visitor implementation to evaluate ade nodes according to ctx and dtype
/// Given a global context containing ade-llo association maps, get data from
/// llo::Sources when possible, otherwise treat native ade::iTensors as zeroes
/// Additionally, Evaluator attempts to get meta-data from llo::FuncWrapper
/// before checking native ade::Functor
template <typename T>
struct Evaluator final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		const ade::Shape& shape = leaf->shape();
		void* ptr = leaf->data();
		age::_GENERATED_DTYPE intype =
			(age::_GENERATED_DTYPE) leaf->type_code();
		size_t n = shape.n_elems();
		std::vector<T> data;
		switch (intype)
		{
			case age::DOUBLE:
				data = TEMPCONVERT(double);
				break;
			case age::FLOAT:
				data = TEMPCONVERT(float);
				break;
			case age::INT8:
				data = TEMPCONVERT(int8_t);
				break;
			case age::INT16:
				data = TEMPCONVERT(int16_t);
				break;
			case age::INT32:
				data = TEMPCONVERT(int32_t);
				break;
			case age::INT64:
				data = TEMPCONVERT(int64_t);
				break;
			case age::UINT8:
				data = TEMPCONVERT(uint8_t);
				break;
			case age::UINT16:
				data = TEMPCONVERT(uint16_t);
				break;
			case age::UINT32:
				data = TEMPCONVERT(uint32_t);
				break;
			case age::UINT64:
				data = TEMPCONVERT(uint64_t);
				break;
			default: logs::fatalf("invalid input type %s",
				age::name_type(intype).c_str());
		}
		out_ = get_tensor(data.data(), shape);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (auto shortcut = dynamic_cast<ShortcutFunctor*>(func))
		{
			out_ = shortcut->evaluate<T>();
			return;
		}

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

		out_ = get_tensor<T>(nullptr, outshape);
		age::typed_exec<T>(opcode, out_, argdata);
	}

	/// Output data evaluated upon visiting node
	TensorT<T> out_;
};

template <typename T>
TensorT<T> ShortcutFunctor::evaluate (void)
{
	size_t nargs = entries_.size();
	DataArgsT<T> argdata(nargs);
	for (size_t i = 0; i < nargs; ++i)
	{
		Evaluator<T> evaler;
		entries_[i].get_tensor()->accept(evaler);
		argdata[i] = DataArg<T>{
			evaler.out_,
			entries_[i].get_coorder(),
			entries_[i].map_io(),
		};
	}

	const ade::Shape& outshape = proxy_root_->shape();
	TensorT<T> out = get_tensor<T>(nullptr, outshape);
	age::typed_exec<T>(opcode_, out, argdata);
	return out;
}

/// Evaluate generic data of tens converted to specified type
template <typename T>
TensorT<T> eval (ade::iTensor* tens)
{
	Evaluator<T> eval;
	tens->accept(eval);
	return eval.out_;
}

/// Evaluate generic data of tens pointer converted to specified type
template <typename T>
TensorT<T> eval (ade::TensptrT tens)
{
	Evaluator<T> eval;
	tens->accept(eval);
	return eval.out_;
}

}

#endif // LLO_EVAL_HPP
