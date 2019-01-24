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

	GenericData evaluate (age::_GENERATED_DTYPE dtype);

private:
	ShortcutFunctor (age::_GENERATED_OPCODE opcode,
		FuncptrT proxy_root, ade::ArgsT& entries) :
		opcode_(opcode), proxy_root_(proxy_root), entries_(entries) {}

	age::_GENERATED_OPCODE opcode_;

	FuncptrT proxy_root_;

	ade::ArgsT entries_;
};

/// Visitor implementation to evaluate ade nodes according to ctx and dtype
/// Given a global context containing ade-llo association maps, get data from
/// llo::Sources when possible, otherwise treat native ade::iTensors as zeroes
/// Additionally, Evaluator attempts to get meta-data from llo::FuncWrapper
/// before checking native ade::Functor
struct Evaluator final : public ade::iTraveler
{
	Evaluator (age::_GENERATED_DTYPE dtype) : dtype_(dtype) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		const char* data = (const char*) leaf->data();
		age::_GENERATED_DTYPE dtype = (age::_GENERATED_DTYPE) leaf->type_code();
		const ade::Shape& shape = leaf->shape();
		out_ = GenericData(shape, dtype_);
		out_.copyover(data, dtype);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (auto shortcut = dynamic_cast<ShortcutFunctor*>(func))
		{
			out_ = shortcut->evaluate(dtype_);
			return;
		}

		age::_GENERATED_OPCODE opcode = (age::_GENERATED_OPCODE)
			func->get_opcode().code_;
		out_ = GenericData(func->shape(), dtype_);

		ade::ArgsT children = func->get_children();
		uint8_t nargs = children.size();
		DataArgsT argdata = DataArgsT(nargs);
		if (func->get_opcode().code_ == age::RAND_BINO)
		{
			if (nargs != 2)
			{
				logs::fatalf("cannot RAND_BINO without exactly 2 arguments: "
					"using %d arguments", nargs);
			}
			Evaluator left_eval(dtype_);
			children[0].get_tensor()->accept(left_eval);
			argdata[0] = {
				left_eval.out_.data_,
				left_eval.out_.shape_,
				children[0].get_coorder(),
				children[0].map_io(),
			};

			Evaluator right_eval(age::DOUBLE);
			children[1].get_tensor()->accept(right_eval);
			argdata[1] = DataArg{
				right_eval.out_.data_,
				right_eval.out_.shape_,
				children[1].get_coorder(),
				children[1].map_io(),
			};
		}
		else
		{
			for (uint8_t i = 0; i < nargs; ++i)
			{
				Evaluator evaler(dtype_);
				children[i].get_tensor()->accept(evaler);
				argdata[i] = DataArg{
					evaler.out_.data_,
					evaler.out_.shape_,
					children[i].get_coorder(),
					children[i].map_io(),
				};
			}
		}

		op_exec(opcode, out_.dtype_, out_.data_.get(), out_.shape_, argdata);
	}

	/// Output data evaluated upon visiting node
	GenericData out_;

private:
	/// Output type when evaluating data
	age::_GENERATED_DTYPE dtype_;
};

/// Evaluate generic data of tens converted to specified dtype
GenericData eval (ade::iTensor* tens, age::_GENERATED_DTYPE dtype);

/// Evaluate generic data of tens pointer converted to specified dtype
GenericData eval (ade::TensptrT tens, age::_GENERATED_DTYPE dtype);

}

#endif // LLO_EVAL_HPP
