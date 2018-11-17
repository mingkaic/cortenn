#include "llo/generated/api.hpp"
#include "llo/generated/grader.hpp"

#ifndef EQNS_ACTIVATIONS_HPP
#define EQNS_ACTIVATIONS_HPP

/// sigmoid function: f(x) = 1/(1+e^-x)
ade::Tensorptr sigmoid (ade::Tensorptr x);

/// tanh function: f(x) = (e^(2*x)+1)/(e^(2*x)-1)
ade::Tensorptr tanh (ade::Tensorptr x);

/// softmax function: f(x) = e^x / sum(e^x)
ade::Tensorptr softmax (ade::Tensorptr x);

#endif
