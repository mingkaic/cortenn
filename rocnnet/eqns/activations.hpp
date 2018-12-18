#include "llo/generated/api.hpp"
#include "llo/generated/grader.hpp"

#ifndef EQNS_ACTIVATIONS_HPP
#define EQNS_ACTIVATIONS_HPP

namespace eqns
{

/// sigmoid function: f(x) = 1/(1+e^-x)
ade::TensptrT sigmoid (ade::TensptrT x);

/// tanh function: f(x) = (e^(2*x)+1)/(e^(2*x)-1)
ade::TensptrT tanh (ade::TensptrT x);

/// softmax function: f(x) = e^x / sum(e^x)
ade::TensptrT softmax (ade::TensptrT x);

}

#endif
