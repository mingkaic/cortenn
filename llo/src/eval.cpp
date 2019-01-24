#include "llo/eval.hpp"

#ifdef LLO_EVAL_HPP

namespace llo
{

GenericData ShortcutFunctor::evaluate (age::_GENERATED_DTYPE dtype)
{
	size_t nargs = entries_.size();
	DataArgsT argdata = DataArgsT(nargs);
	for (size_t i = 0; i < nargs; ++i)
	{
		Evaluator evaler(dtype);
		entries_[i].get_tensor()->accept(evaler);
		argdata[i] = DataArg{
			evaler.out_.data_,
			evaler.out_.shape_,
			entries_[i].get_coorder(),
			entries_[i].map_io(),
		};
	}

	GenericData out(proxy_root_->shape(), dtype);
	op_exec(opcode_, dtype, out.data_.get(), out.shape_, argdata);
	return out;
}

GenericData eval (ade::iTensor* tens, age::_GENERATED_DTYPE dtype)
{
	Evaluator eval(dtype);
	tens->accept(eval);
	return eval.out_;
}

GenericData eval (ade::TensptrT tens, age::_GENERATED_DTYPE dtype)
{
	Evaluator eval(dtype);
	tens->accept(eval);
	return eval.out_;
}

}

#endif
