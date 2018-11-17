#include "llo/operator.hpp"
#include "llo/generated/api.hpp"

#include "rocnnet/layr/ilayer.hpp"

#ifndef LAYR_CONV_LAYER_HPP
#define LAYR_CONV_LAYER_HPP

struct ConvLayer : public iLayer
{
	ConvLayer (std::pair<uint8_t,uint8_t> filter_hw,
		uint8_t in_ncol, uint8_t out_ncol, std::string label) :
		iLayer(label)
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

	virtual ~ConvLayer (void) {}

	ade::Tensorptr operator () (ade::Tensorptr input)
	{
		return age::add(age::convolute(input, ade::Tensorptr(weight_)),
			ade::Tensorptr(bias_));
	}

	std::vector<llo::VarptrT> get_variables (void) const override
	{
		return {weight_, bias_};
	}

protected:
	llo::VarptrT weight_;

	llo::VarptrT bias_;
};

#endif // LAYR_CONV_LAYER_HPP
