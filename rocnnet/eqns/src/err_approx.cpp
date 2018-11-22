#include "llo/generated/api.hpp"
#include "llo/shear.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifdef EQNS_ERR_APPROX_HPP

DeltasT sgd (ade::TensptrT& root, VariablesT leaves,
	double learning_rate)
{
	DeltasT errs;
	for (llo::VarptrT& leaf : leaves)
	{
		auto gres = llo::derive(root, ade::TensptrT(leaf));
		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		ade::Shape gshape = gres->shape();
		errs.emplace(leaf.get(),
			age::sub(ade::TensptrT(leaf),
				age::mul(gres,
					llo::data(learning_rate, gshape, "learning_rate"))
			));
	}
	return errs;
}

DeltasT rms_momentum (ade::TensptrT& root, VariablesT leaves,
	double learning_rate, double discount_factor, double epsilon)
{
	DeltasT errs;
	for (llo::VarptrT& leaf : leaves)
	{
		auto gres = llo::derive(root, ade::TensptrT(leaf));
		// given root = f, err(x) ~ x - (η * df(x)) / (sqrt(ε + momentum)),
		// where η is the learning rate, and ε is epsilon

		// upkeep additional hidden variable momentum: starting with value 1
		// given root = f, err(momentum) ~ χ * momentum + (1 - χ) * df(x) ^ 2,
		// where χ is discount_factor
		ade::Shape shape = leaf->shape();
		llo::VarptrT momentum = llo::data<double>(1, shape, "momentum");
		ade::TensptrT discount_node = llo::data(discount_factor, shape, "discount");
		ade::TensptrT datcount_node = llo::data(1.0 - discount_factor, shape, "1-discount");

		errs.emplace(momentum.get(),
			age::add(
				age::mul(discount_node, momentum),
				age::prod({datcount_node, gres, gres})
			));

		errs.emplace(leaf.get(),
			age::sub(ade::TensptrT(leaf),
				age::div(
					age::mul(
						ade::TensptrT(gres),
						llo::data(learning_rate, shape, "learning_rate")
					),
					age::add(age::sqrt(momentum), llo::data(epsilon, shape, "epsilon"))
				)
			));
	}
	return errs;
}

#endif
