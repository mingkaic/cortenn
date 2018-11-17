#include <unordered_map>

#include "llo/data.hpp"

#ifndef EQNS_ERR_APPROX_HPP
#define EQNS_ERR_APPROX_HPP

using VariablesT = std::vector<llo::VarptrT>;

using DeltasT = std::unordered_map<llo::Variable*,ade::Tensorptr>;

// approximate error of sources given error of root
using ApproxFuncT = std::function<DeltasT(ade::Tensorptr&,VariablesT)>;

// Stochastic Gradient Descent Approximation
DeltasT sgd (ade::Tensorptr& root, VariablesT leaves,
	double learning_rate);

// Momentum-based Root Mean Square Approximation
DeltasT rms_momentum (ade::Tensorptr& root, VariablesT leaves,
	double learning_rate, double discount_factor, double epsilon);

#endif // EQNS_ERR_APPROX_HPP
