#include "llo/eval.hpp"

#ifdef LLO_EVAL_HPP

namespace llo
{

GenericData eval (ade::TensptrT tens, age::_GENERATED_DTYPE dtype)
{
	Evaluator eval(dtype);
	tens->accept(eval);
	return eval.out_;
}

}

#endif
