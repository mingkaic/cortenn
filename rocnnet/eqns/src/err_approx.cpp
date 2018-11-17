#include "llo/generated/api.hpp"
#include "llo/shear.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifdef EQNS_ERR_APPROX_HPP

DeltasT sgd (ade::Tensorptr& root, VariablesT leaves,
	double learning_rate)
{
	DeltasT errs;
	for (llo::VarptrT& leaf : leaves)
	{
		auto der = age::derive(root, leaf.get());
		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		ade::Tensorptr gres = llo::zero_prune(der);
		ade::Shape gshape = gres->shape();
		errs.emplace(leaf.get(),
			age::sub(ade::Tensorptr(leaf),
				age::mul(gres,
					llo::data(learning_rate, gshape, "learning_rate"))
			));
	}
	return errs;
}

DeltasT rms_momentum (ade::Tensorptr& root, VariablesT leaves,
	double learning_rate, double discount_factor, double epsilon)
{
	DeltasT errs;
	for (llo::VarptrT& leaf : leaves)
	{
		auto der = age::derive(root, leaf.get());
		// given root = f, err(x) ~ x - (η * df(x)) / (sqrt(ε + momentum)),
		// where η is the learning rate, and ε is epsilon
		ade::Tensorptr gres = llo::zero_prune(der);

		// upkeep additional hidden variable momentum: starting with value 1
		// given root = f, err(momentum) ~ χ * momentum + (1 - χ) * df(x) ^ 2,
		// where χ is discount_factor
		ade::Shape shape = leaf->shape();
		llo::Variable* momentum = llo::data<double>(1, shape, "momentum");
		ade::Tensorptr discount_node = llo::data(discount_factor, shape, "discount");
		ade::Tensorptr datcount_node = llo::data(1.0 - discount_factor, shape, "1-discount");

		errs.emplace(momentum,
			age::add(
				age::mul(discount_node, momentum),
				age::prod({datcount_node, gres, gres})
			));

		errs.emplace(leaf.get(),
			age::sub(ade::Tensorptr(leaf),
				age::div(
					age::mul(
						ade::Tensorptr(gres),
						llo::data(learning_rate, shape, "learning_rate")
					),
					age::add(age::sqrt(momentum), llo::data(epsilon, shape, "epsilon"))
				)
			));
	}
	return errs;
}

#endif
