#include "llo/generated/api.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_FC_LAYER_HPP
#define MODL_FC_LAYER_HPP

namespace modl
{

const std::string weight_fmt = "weight_%d";
const std::string bias_fmt = "bias_%d";

struct FCLayer final : public iMarshalSet
{
	FCLayer (std::vector<uint8_t> n_inputs, uint8_t n_output,
		std::string label) : iMarshalSet(label)
	{
		size_t n = n_inputs.size();
		if (n == 0)
		{
			logs::fatal("cannot create FCLayer with no inputs");
		}
		for (size_t i = 0; i < n; ++i)
		{
			ade::Shape shape({n_output, n_inputs[i]});
			size_t ndata = shape.n_elems();

			double bound = 1.0 / std::sqrt(n_inputs[i]);
			std::uniform_real_distribution<double> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(llo::get_engine());
			};
			std::vector<double> data(ndata);
			std::generate(data.begin(), data.end(), gen);

			llo::VarptrT weight(llo::get_variable(data, shape,
				fmts::sprintf(weight_fmt, i)));
			llo::VarptrT bias(llo::data<double>(0, ade::Shape({n_output}),
				fmts::sprintf(bias_fmt, i)));
			weight_bias_.push_back({
				std::make_shared<MarshalVar>(weight),
				std::make_shared<MarshalVar>(bias),
			});
		}
	}

	FCLayer (const FCLayer& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	FCLayer& operator = (const FCLayer& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	FCLayer (FCLayer&& other) = default;

	FCLayer& operator = (FCLayer&& other) = default;


	ade::TensptrT operator () (ade::TensT inputs)
	{
		size_t n = inputs.size();
		if (n != weight_bias_.size())
		{
			logs::fatalf("number of inputs must be exactly %d", n);
		}
		ade::TensT args;
		for (size_t i = 0; i < n; ++i)
		{
			ade::DimT cdim = inputs[i]->shape().at(1);
			args.push_back(age::matmul(
				inputs[i], weight_bias_[i].weight_->var_));
			args.push_back(age::extend(
				weight_bias_[i].bias_->var_, 1, {cdim}));
		}
		return age::sum(args);
	}

	uint8_t get_ninput (void) const
	{
		return weight_bias_[0].weight_->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_bias_[0].weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out;
		for (const LoneLayer& wbpair : weight_bias_)
		{
			out.push_back(wbpair.weight_);
			out.push_back(wbpair.bias_);
		}
		return out;
	}

private:
	struct LoneLayer
	{
		MarVarsptrT weight_;
		MarVarsptrT bias_;
	};

	std::vector<LoneLayer> weight_bias_;

	void copy_helper (const FCLayer& other)
	{
		weight_bias_.clear();
		for (const LoneLayer& opair : other.weight_bias_)
		{
			llo::VarptrT weight(new llo::Variable(*(opair.weight_->var_)));
			llo::VarptrT bias(new llo::Variable(*(opair.bias_->var_)));
			weight_bias_.push_back({
				std::make_shared<MarshalVar>(weight),
				std::make_shared<MarshalVar>(bias),
			});
		}
	}
};

using FCptrT = std::shared_ptr<FCLayer>;

}

#endif // MODL_FC_LAYER_HPP
