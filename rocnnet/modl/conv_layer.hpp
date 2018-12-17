#include "llo/data.hpp"
#include "llo/operator.hpp"
#include "llo/generated/api.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_CONV_LAYER_HPP
#define MODL_CONV_LAYER_HPP

struct ConvLayer final : public iMarshalSet
{
	ConvLayer (std::pair<uint8_t,uint8_t> filter_hw, uint8_t in_ncol,
		uint8_t out_ncol, std::string label) : iMarshalSet(label)
	{
		ade::Shape shape({filter_hw.first,
			filter_hw.second, in_ncol, out_ncol});
		size_t ndata = shape.n_elems();

		size_t input_size = filter_hw.first * filter_hw.second * in_ncol;
		double bound = 1.0 / std::sqrt(input_size);
		std::uniform_real_distribution<double> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(llo::get_engine());
		};
		std::vector<double> data(ndata);
		std::generate(data.begin(), data.end(), gen);

		weight_ = std::make_shared<MarshalVar>(
			llo::get_variable(data, shape, "weight"));
		bias_ = std::make_shared<MarshalVar>(
			llo::data<double>(0, ade::Shape({out_ncol}), "bias"));
	}

	ade::TensptrT operator () (ade::TensptrT input)
	{
		return age::add(age::convolute(input,
			ade::TensptrT(weight_->var_)), ade::TensptrT(bias_->var_));
	}

	uint8_t get_ninput (void) const
	{
		return weight_->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		return {weight_, bias_};
	}

protected:
	std::string label_;

	std::shared_ptr<MarshalVar> weight_;

	std::shared_ptr<MarshalVar> bias_;
};

#endif // MODL_CONV_LAYER_HPP
