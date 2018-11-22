#include "rocnnet/modl/fc_layer.hpp"
#include "rocnnet/modl/rbm.hpp"

using PretrainsT = std::vector<DeltasNCostT>;

struct DBTrainer final
{
	DBTrainer (uint8_t n_input, std::vector<uint8_t> hiddens, std::string label) :
		label_("db_" + label), n_input_(n_input), n_output_(hiddens.back()),
        log_layer_({n_input}, hiddens.back(), label + ":logres")
	{
        if (hiddens.empty())
        {
            err::fatal("cannot db train with no hiddens");
        }
        for (size_t level = 0, n = hiddens.size(); level < n; ++level)
        {
            uint8_t n_out = hiddens[level];
            layers_.push_back(RBM(n_input, n_out,
                err::sprintf("rbm_%d", level)));
            n_input = n_out;
        }
	}

	DBTrainer (const DBTrainer& other, std::string label_prefix = "copied_")
	{
		copy_helper(other, label_prefix);
	}

	DBTrainer& operator = (const DBTrainer& other)
	{
		if (this != &other)
		{
			copy_helper(other, "copied_");
		}
		return *this;
	}

	DBTrainer (DBTrainer&& other) = default;

	DBTrainer& operator = (DBTrainer&& other) = default;


    // input of shape <n_input, n_batch>
	ade::TensptrT prop_up (ade::TensptrT input)
	{
        // sanity check
        const ade::Shape& in_shape = input->shape();
        if (in_shape.at(0) != n_input_)
        {
            err::fatalf("cannot dbn with input shape %s against n_input %d",
                in_shape.to_string().c_str(), n_input_);
        }
        ade::TensptrT output = input;
        for (RBM& h : layers_)
        {
            output = h.prop_up(output);
        }
        return softmax(log_layer_({ output }));
	}

    PretrainsT pretraining_functions (llo::VarptrT input,
		double learning_rate = 1e-3, size_t n_cont_div = 10)
    {
        ade::TensptrT input_node = input;
        PretrainsT pt_updates;
        for (RBM& h : layers_)
        {
            pt_updates.push_back(h.train(input_node, nullptr, learning_rate, n_cont_div));
            input_node = h.prop_up(input_node);
        }
        return pt_updates;
    }

	DeltasNCostT build_finetune_functions (llo::VarptrT train_in,
		llo::VarptrT train_out, double learning_rate = 1e-3)
    {
        ade::TensptrT out_dist = prop_up(ade::TensptrT(train_in));
        ade::TensptrT finetune_cost = - age::reduce_mean(age::log(out_dist));

        finetune_cost->set_label("finetune_cost(" + finetune_cost->get_label() + ")");
        nnet::iconnector<double>* ft_cost_icon = static_cast<nnet::iconnector<double>*>(finetune_cost.get());

        ade::TensptrT temp_diff = age::sub(out_dist, ade::TensptrT(train_out));
        ade::TensptrT error = age::reduce_mean(age::pow(temp_diff, llo::data(2, temp_diff->shape(), "2"));

        std::vector<llo::VarptrT> gparams = this->get_variables();
        DeltasT errs;
        for (llo::VarptrT& gp : gparams)
        {
            errs.emplace(gp.get(), age::sub(ade::TensptrT(gp),
                age::mul(llo::data(learning_rate, gp->shape(), "learning_rate"),
                    llo::derive(finetune_cost, gp))));
        }

        return {errs, error};
    }

	std::vector<llo::VarptrT> get_variables (void) const
	{
        std::vector<llo::VarptrT> vars;
        for (RBM& h : layers_)
        {
            std::vector<llo::VarptrT> temp = h.get_variables();
            vars.insert(vars.end(), temp.begin(), temp.end());
        }
        std::vector<llo::VarptrT> temp = log_layer_.get_variables();
        vars.insert(vars.end(), temp.begin(), temp.end());
        return vars;
	}

	uint8_t get_ninput (void) const
	{
		return n_input_;
	}

	uint8_t get_noutput (void) const
	{
        return n_output_;
	}

	void parse_from (pbm::LoadVecsT labels)
	{
		pbm::LoadVecsT relevant;
		std::copy_if(labels.begin(), labels.end(), std::back_inserter(relevant),
			[&](pbm::LoadTensT& pairs)
			{
				return pairs.second.size() > 0 &&
					this->label_ == pairs.second.front();
			});
		for (pbm::LoadTensT& pairs : relevant)
		{
			pairs.second.pop_front();
		}
		for (RBM& rlayer : layers_)
		{
            rlayer.parse_from(relevant);
		}
	}

private:
	void copy_helper (const DBTrainer& other, std::string prefix)
	{
		label_ = prefix + other.label_;
		n_input_ = other.n_input_;
		n_output_ = other.n_output_;
        layers_ = other.layers_;
        log_layer_ = other.log_layer_;
	}

	std::string label_;

	uint8_t n_input_;

    uint8_t n_output_;

    std::vector<RBM> layers_;

	FCLayer log_layer_;
};
