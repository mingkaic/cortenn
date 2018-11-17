#include "rocnnet/eqns/activations.hpp"

#ifdef EQNS_ACTIVATIONS_HPP

ade::Tensorptr sigmoid (ade::Tensorptr x)
{
	ade::Shape shape = x->shape();
	auto denom = age::add(age::data(1, shape), age::exp(age::neg(x)));
	return age::div(age::data(1, shape), denom);
}

ade::Tensorptr tanh (ade::Tensorptr x)
{
	ade::Shape shape = x->shape();
	auto expxx = age::exp(age::add(x, x));
	auto num = age::add(expxx, age::data(1, shape));
	auto denom = age::sub(expxx, age::data(1, shape));
	return age::div(num, denom);
}

ade::Tensorptr softmax (ade::Tensorptr x)
{
	auto num = age::exp(x);
	auto denom = age::reduce_sum(num);
	return age::div(num, denom);
}

#endif
