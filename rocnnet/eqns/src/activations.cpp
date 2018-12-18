#include "rocnnet/eqns/activations.hpp"

#ifdef EQNS_ACTIVATIONS_HPP

namespace eqns
{

ade::TensptrT sigmoid (ade::TensptrT x)
{
	ade::Shape shape = x->shape();
	auto denom = age::add(ade::TensptrT(age::data(1, shape)),
		age::exp(age::neg(x)));
	return age::div(ade::TensptrT(age::data(1, shape)), denom);
}

ade::TensptrT tanh (ade::TensptrT x)
{
	ade::Shape shape = x->shape();
	auto expxx = age::exp(age::add(x, x));
	auto num = age::add(expxx, ade::TensptrT(age::data(1, shape)));
	auto denom = age::sub(expxx, ade::TensptrT(age::data(1, shape)));
	return age::div(num, denom);
}

ade::TensptrT softmax (ade::TensptrT x)
{
	auto num = age::exp(x);
	auto denom = age::reduce_sum(num);
	return age::div(num, denom);
}

}

#endif
