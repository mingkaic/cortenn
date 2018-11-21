#include <functional>
#include <memory>

#include "llo/operator.hpp"

using DeltasNCostT = std::pair<DeltasT,ade::Tensorptr>;

ade::Tensorptr one_binom (ade::Tensorptr a)
{
    return age::rand_bino(llo::data<double>(1.0, a->shape(), "1"), a);
}

ade::Tensorptr extended_add (ade::Tensorptr bigger, ade::Tensorptr smaller)
{
    ade::DimT cdim = bigger->shape().at(1);
    return age::add(bigger, age::extend(ade::Tensorptr(smaller), 1, {cdim}));
}

struct RBM final : public Unmarshaler
{
	RBM (uint8_t n_input, uint8_t n_hidden, std::string label) :
		label_(label), n_input_(n_input), n_hidden_(n_hidden)
	{
        ade::Shape weight_shape({n_hidden, n_input});
        size_t nw = weight_shape.n_elems();

        double bound = 4 * std::sqrt(6.0 / (n_hidden + n_input));
        std::uniform_real_distribution<double> dist(-bound, bound);
        auto gen = [&dist]()
        {
            return dist(llo::get_engine());
        };
        std::vector<double> wdata(nw);
        std::generate(wdata.begin(), wdata.end(), gen);

        weight_ = llo::get_variable(wdata, weight_shape, "weight");
		hbias_ = llo::data<double>(0, ade::Shape({n_hidden}), "hbias");
		vbias_ = llo::data<double>(0, ade::Shape({n_input}), "vbias");
	}

	RBM (const RBM& other, std::string label_prefix = "copied_")
	{
		copy_helper(other, label_prefix);
	}

	RBM& operator = (const RBM& other)
	{
		if (this != &other)
		{
			copy_helper(other, "copied_");
		}
		return *this;
	}

	RBM (RBM&& other)
	{
		move_helper(std::move(other));
	}

	RBM& operator = (RBM&& other)
	{
		if (this != &other)
		{
			label_ = other.label_;
			move_helper(std::move(other));
		}
		return *this;
	}


    // input of shape <n_input, n_batch>
	ade::Tensorptr prop_up (ade::Tensorptr input)
	{
        // prop forward
        // weight is <n_hidden, n_input>
        // in is <n_input, ?>
        // out = in @ weight, so out is <n_hidden, ?>
        ade::Tensorptr weighed = age::matmul(input, weight_);
        ade::Tensorptr pre_nl = extended_add(weighted, hbias_);
        return sigmoid(pre_nl);
	}

    // input of shape <n_hidden, n_batch>
	ade::Tensorptr prop_down (ade::Tensorptr hidden)
	{
        // weight is <n_hidden, n_input>
        // in is <n_hidden, ?>
        // out = in @ weight.T, so out is <n_input, ?>
        ade::DimT cdim = input->shape().at(1);
        ade::Tensorptr weighed = age::matmul(input, age::transpose(weight_));
        ade::Tensorptr pre_nl = extended_add(weighted, vbias_);
        return sigmoid(pre_nl);
	}

    // recreate input using hidden distribution
    // output shape of input->shape()
	ade::Tensorptr reconstruct_visible (ade::Tensorptr input)
	{
        ade::Tensorptr hidden_dist = prop_up(input);
        ade::Tensorptr hidden_sample = one_binom(hidden_dist);
        return prop_down(hidden_sample);
	}

	ade::Tensorptr reconstruct_hidden (ade::Tensorptr hidden)
	{
        ade::Tensorptr visible_dist = prop_down(hidden);
        ade::Tensorptr visible_sample = one_binom(visible_dist);
        return prop_up(visible_sample);
	}


    // todo: move this somewhere else
    // input a 2-D vector of shape <n_input, n_batch>
	DeltasNCostT train (ade::Tensorptr input,
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
        // otherwise use Persistent CD (initialize from the old state of the chain)
        else
        {
            chain_it = persistent;
        }
        ade::Tensorptr chain(chain_it);

        std::shared_ptr<iTensor> final_visible_dist;
        for (size_t i = 0; i < n_cont_div; i++)
        {
            ade::Tensorptr hidden_dist = reconstruct_hidden(chain);

            // use operational optimization to recover presig and vis nodes
            ade::Tensorptr weighed = age::matmul(chain_it, age::transpose(weight_));
            ade::Tensorptr presig_vis = extended_add(weighted, vbias_);
            final_visible_dist = age::sigmoid(presig_vis);

            chain_it = one_binom(hidden_dist);
        }
        ade::Tensorptr chain_end = one_binom(final_visible_dist);

        ade::Tensorptr cost = age::sub(
            age::reduce_mean(free_energy(input)),
            age::reduce_mean(free_energy(chain_end)));

        ade::Tensorptr dW = age::derive(cost, weight_.get());
        ade::Tensorptr dhb = age::derive(cost, hbias_.get());
        ade::Tensorptr dvb = age::derive(cost, vbias_.get());

    	DeltasT errs;
        errs.emplace(weight_.get(), age::sub(ade::Tensorptr(weight_),
            age::mul(llo::data(learning_rate, dW->shape(), "learning_rate"), dW)));
        errs.emplace(hbias_.get(), age::sub(ade::Tensorptr(hbias_),
            age::mul(llo::data(learning_rate, dhb->shape(), "learning_rate"), dhb)));
        errs.emplace(vbias_.get(), age::sub(ade::Tensorptr(vbias_),
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


	std::vector<llo::VarptrT> get_variables (void) const
	{
        return {weight_, hbias_, vbias_};
	}

	uint8_t get_ninput (void) const
	{
		return n_input_;
	}

	uint8_t get_noutput (void) const
	{
		return n_hidden_;
	}

	void parse_from (pbm::LoadVecsT labels)
	{
		//
	}

private:
	void copy_helper (const RBM& other, std::string prefix)
	{
		label_ = prefix + other.label_;
		n_input_ = other.n_input_;
		n_hidden_ = other.n_hidden_;
        weight_ = llo::VarptrT(new llo::Variable(*other.weight_));
        hbias_ = llo::VarptrT(new llo::Variable(*other.hbias_));
        vbias_ = llo::VarptrT(new llo::Variable(*other.vbias_));
	}

	void move_helper (RBM&& other)
	{
		label_ = std::move(other.label_);
		n_input_ = std::move(other.n_input_);
		n_hidden_ = std::move(other.n_hidden_);
        weight_ = std::move(other.weight_);
        hbias_ = std::move(other.hbias_);
        vbias_ = std::move(other.vbias_);
	}

    ade::Tensorptr free_energy (ade::Tensorptr sample)
    {
        ade::Tensorptr vbias_term = age::matmul(sample, age::transpose(ade::Tensorptr(vbias_)));
        // <x, y> @ <z, x> + z -> <z, y>
        ade::Tensorptr weighed = age::matmul(sample, ade::Tensorptr(weight_));
        ade::Tensorptr wx_b = extended_add(weighed, ade::Tensorptr(hbias_));
        ade::Tensorptr hidden_term = age::reduce_sum(
            age::transpose(age::log(
                age::add(llo::data(1.0, wx_b->shape(), "1"), age::exp(wx_b))
            )), 1);
        return ade::neg(ade::add(hidden_term, vbias_term));
    }

    ade::Tensorptr get_pseudo_likelihood_cost (ade::Tensorptr input)
    {
        const ade::Shape& shape = input->shape();
        std::vector<double> zeros(shape.n_elems(), 0);
        zeros[0] = 1;
        ade::Tensorptr one_i = llo::get_variable(zeros, shape);

        ade::Tensorptr xi = age::round(input); // xi = [0|1, ...]
        ade::Tensorptr xi_flip = age::sub(one_i, xi);

        ade::Tensorptr fe_xi = free_energy(xi);
        ade::Tensorptr fe_xi_flip = free_energy(xi_flip);

        return age::reduce_mean(age::mul(
            llo::data<double>(n_input_, fe_xi->shape(), "n_input"),
            age::log(sigmoid(age::sub(fe_xi_flip, fe_xi)))));
    }

	ade::Tensorptr get_reconstruction_cost (ade::Tensorptr input, ade::Tensorptr visible_dist)
    {
        ade::Tensorptr p_success = age::mul(input, age::log(visible_dist));
        ade::Tensorptr p_not = age::mul(
            age::sub(llo::data(1, input->shape(), "1"), input),
            age::log(age::sub(llo::data(1, visible_dist->shape(), "1"), visible_dist)));
        return age::reduce_mean(
            age::reduce_sum(
                age::transpose(age::add(p_success, p_not)), 1));
    }

	std::string label_;

	uint8_t n_input_;

    uint8_t n_hidden_;

    llo::VarptrT weight_;

	llo::VarptrT hbias_;

	llo::VarptrT vbias_;
};
