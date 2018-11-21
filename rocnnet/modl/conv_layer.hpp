#include "llo/operator.hpp"
#include "llo/generated/api.hpp"

#ifndef MODL_CONV_LAYER_HPP
#define MODL_CONV_LAYER_HPP

struct ConvLayer final
{
	ConvLayer (std::pair<uint8_t,uint8_t> filter_hw,
		uint8_t in_ncol, uint8_t out_ncol, std::string label) :
		label_("conv_" + label)
	{
		ade::Shape shape({filter_hw.first, filter_hw.second, in_ncol, out_ncol});
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

		weight_ = llo::VarptrT(
			llo::get_variable(data, shape, "weight"));
		bias_ = llo::VarptrT(
			llo::data<double>(0, ade::Shape({out_ncol}), "bias"));
	}

	ade::Tensorptr operator () (ade::Tensorptr input)
	{
		return age::add(age::convolute(input, ade::Tensorptr(weight_)),
			ade::Tensorptr(bias_));
	}

	std::vector<llo::VarptrT> get_variables (void) const
	{
		return {weight_, bias_};
	}

	uint8_t get_ninput (void) const
	{
		return weight_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_->shape().at(0);
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
		weight_->parse_from(relevant);
		bias_->parse_from(relevant);
	}

protected:
	std::string label_;

	llo::VarptrT weight_;

	llo::VarptrT bias_;
};

#endif // MODL_CONV_LAYER_HPP
