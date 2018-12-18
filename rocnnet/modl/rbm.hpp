#include <functional>
#include <memory>

#include "llo/operator.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_RBM_HPP
#define MODL_RBM_HPP

namespace modl
{

using DeltasNCostT = std::pair<DeltasT,ade::TensptrT>;

ade::TensptrT one_binom (ade::TensptrT a)
{
	return age::rand_bino(llo::data<double>(1.0, a->shape(), "1"), a);
}

ade::TensptrT extended_add (ade::TensptrT bigger, ade::TensptrT smaller)
{
	ade::DimT cdim = bigger->shape().at(1);
	return age::add(bigger, age::extend(ade::TensptrT(smaller), 1, {cdim}));
}

struct RBM final : public iMarshalSet
{
	RBM (uint8_t n_input, uint8_t n_hidden, std::string label) :
		iMarshalSet(label), n_input_(n_input), n_hidden_(n_hidden)
	{
		ade::Shape shape({n_hidden, n_input});
		size_t nw = shape.n_elems();

		double bound = 4 * std::sqrt(6.0 / (n_hidden + n_input));
		std::uniform_real_distribution<double> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(llo::get_engine());
		};
		std::vector<double> wdata(nw);
		std::generate(wdata.begin(), wdata.end(), gen);

		weight_ = MarVarsptrT(
			llo::get_variable(wdata, shape, "weight"));
		hbias_ = MarVarsptrT(
			llo::data<double>(0, ade::Shape({n_hidden}), "hbias"));
		vbias_ = MarVarsptrT(
			llo::data<double>(0, ade::Shape({n_input}), "vbias"));
	}

	RBM (const RBM& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	RBM& operator = (const RBM& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	RBM (RBM&& other) = default;

	RBM& operator = (RBM&& other) = default;


	// input of shape <n_input, n_batch>
	ade::TensptrT prop_up (ade::TensptrT input)
	{
		// prop forward
		// weight is <n_hidden, n_input>
		// in is <n_input, ?>
		// out = in @ weight, so out is <n_hidden, ?>
		ade::TensptrT weighed = age::matmul(input, weight_->var_);
		ade::TensptrT pre_nl = extended_add(weighted, hbias_->var_);
		return eqns::sigmoid(pre_nl);
	}

	// input of shape <n_hidden, n_batch>
	ade::TensptrT prop_down (ade::TensptrT hidden)
	{
		// weight is <n_hidden, n_input>
		// in is <n_hidden, ?>
		// out = in @ weight.T, so out is <n_input, ?>
		ade::DimT cdim = input->shape().at(1);
		ade::TensptrT weighed = age::matmul(input,
			age::transpose(weight_->var_));
		ade::TensptrT pre_nl = extended_add(weighted, vbias_->var_);
		return eqns::sigmoid(pre_nl);
	}

	// recreate input using hidden distribution
	// output shape of input->shape()
	ade::TensptrT reconstruct_visible (ade::TensptrT input)
	{
		ade::TensptrT hidden_dist = prop_up(input);
		ade::TensptrT hidden_sample = one_binom(hidden_dist);
		return prop_down(hidden_sample);
	}

	ade::TensptrT reconstruct_hidden (ade::TensptrT hidden)
	{
		ade::TensptrT visible_dist = prop_down(hidden);
		ade::TensptrT visible_sample = one_binom(visible_dist);
		return prop_up(visible_sample);
	}


	// todo: move this somewhere else
	// input a 2-D vector of shape <n_input, n_batch>
	DeltasNCostT train (ade::TensptrT input,
		llo::VarptrT persistent = nullptr,
		double learning_rate = 1e-3,
		size_t n_cont_div = 1)
	{
		std::shared_ptr<iTensor> chain_it;
		// if persistent not available use Contrastive Divergence (CD)
		if (nullptr == persistent)
		{
			llo::VarptrT hidden_dist = prop_up(input);
			chain_it = one_binom(hidden_dist);
		}
		// otherwise use Persistent CD
		// (initialize from the old state of the chain)
		else
		{
			chain_it = persistent;
		}
		ade::TensptrT chain(chain_it);

		std::shared_ptr<iTensor> final_visible_dist;
		for (size_t i = 0; i < n_cont_div; i++)
		{
			ade::TensptrT hidden_dist = reconstruct_hidden(chain);

			// use operational optimization to recover presig and vis nodes
			ade::TensptrT weighed = age::matmul(chain_it,
				age::transpose(weight_->var_));
			ade::TensptrT presig_vis = extended_add(weighted, vbias_);
			final_visible_dist = eqns::sigmoid(presig_vis);

			chain_it = one_binom(hidden_dist);
		}
		ade::TensptrT chain_end = one_binom(final_visible_dist);

		ade::TensptrT cost = age::sub(
			age::reduce_mean(free_energy(input)),
			age::reduce_mean(free_energy(chain_end)));

		ade::TensptrT dW = llo::derive(cost, weight_->var_);
		ade::TensptrT dhb = llo::derive(cost, hbias_->var_);
		ade::TensptrT dvb = llo::derive(cost, vbias_->var_);

		DeltasT errs;
		errs.emplace(weight_->var_.get(), age::sub(ade::TensptrT(weight_->var_),
			age::mul(llo::data(learning_rate, dW->shape(), "learning_rate"), dW)));
		errs.emplace(hbias_->var_.get(), age::sub(ade::TensptrT(hbias_->var_),
			age::mul(llo::data(learning_rate, dhb->shape(), "learning_rate"), dhb)));
		errs.emplace(vbias_->var_.get(), age::sub(ade::TensptrT(vbias_->var_),
			age::mul(llo::data(learning_rate, dvb->shape(), "learning_rate"), dvb)));

		std::shared_ptr<iTensor> monitoring_cost;
		if (nullptr == persistent)
		{
			// reconstruction cost
			monitoring_cost = get_reconstruction_cost(input, final_visible_dist);
		}
		else
		{
			// pseudo-likelihood
			errs.emplace(persistent.get(), chain_it);
			monitoring_cost = get_pseudo_likelihood_cost(input);
		}

		return {errs, monitoring_cost};
	}


	uint8_t get_ninput (void) const
	{
		return n_input_;
	}

	uint8_t get_noutput (void) const
	{
		return n_hidden_;
	}

	MarsarrT get_subs (void) const override
	{
		return {weight_, hbias_, vbias_};
	}

private:
	void copy_helper (const RBM& other)
	{
		n_input_ = other.n_input_;
		n_hidden_ = other.n_hidden_;
		weight_ = std::make_shared<MarshalVar>(
			new llo::Variable(*other.weight_->var_));
		hbias_ = std::make_shared<MarshalVar>(
			new llo::Variable(*other.hbias_->var_));
		vbias_ = std::make_shared<MarshalVar>(
			new llo::Variable(*other.vbias_->var_));
	}

	ade::TensptrT free_energy (ade::TensptrT sample)
	{
		ade::TensptrT vbias_term = age::matmul(sample, age::transpose(ade::TensptrT(vbias_)));
		// <x, y> @ <z, x> + z -> <z, y>
		ade::TensptrT weighed = age::matmul(sample, ade::TensptrT(weight_));
		ade::TensptrT wx_b = extended_add(weighed, ade::TensptrT(hbias_));
		ade::TensptrT hidden_term = age::reduce_sum(
			age::transpose(age::log(
				age::add(llo::data(1.0, wx_b->shape(), "1"), age::exp(wx_b))
			)), 1);
		return ade::neg(ade::add(hidden_term, vbias_term));
	}

	ade::TensptrT get_pseudo_likelihood_cost (ade::TensptrT input)
	{
		const ade::Shape& shape = input->shape();
		std::vector<double> zeros(shape.n_elems(), 0);
		zeros[0] = 1;
		ade::TensptrT one_i = llo::get_variable(zeros, shape);

		ade::TensptrT xi = age::round(input); // xi = [0|1, ...]
		ade::TensptrT xi_flip = age::sub(one_i, xi);

		ade::TensptrT fe_xi = free_energy(xi);
		ade::TensptrT fe_xi_flip = free_energy(xi_flip);

		return age::reduce_mean(age::mul(
			llo::data<double>(n_input_, fe_xi->shape(), "n_input"),
			age::log(eqns::sigmoid(age::sub(fe_xi_flip, fe_xi)))));
	}

	ade::TensptrT get_reconstruction_cost (ade::TensptrT input, ade::TensptrT visible_dist)
	{
		ade::TensptrT p_success = age::mul(input, age::log(visible_dist));
		ade::TensptrT p_not = age::mul(
			age::sub(llo::data(1, input->shape(), "1"), input),
			age::log(age::sub(llo::data(1, visible_dist->shape(), "1"), visible_dist)));
		return age::reduce_mean(
			age::reduce_sum(
				age::transpose(age::add(p_success, p_not)), 1));
	}

	uint8_t n_input_;

	uint8_t n_hidden_;

	MarVarsptrT weight_;

	MarVarsptrT hbias_;

	MarVarsptrT vbias_;
};

}

#endif // MODL_RBM_HPP
