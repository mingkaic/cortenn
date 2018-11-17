#include "llo/operator.hpp"

#include "rocnnet/layr/ilayer.hpp"

#ifndef LAYR_FC_LAYER_HPP
#define LAYR_FC_LAYER_HPP

struct FCLayer : public iLayer
{
	FCLayer (std::vector<uint8_t> n_inputs, uint8_t n_output, std::string label) :
		iLayer(label)
	{
		size_t n = n_inputs.size();
		if (n == 0)
		{
			err::fatal("cannot create FCLayer with no inputs");
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

			llo::VarptrT weight(
				llo::get_variable(data, shape, "weight"));
			llo::VarptrT bias(
				llo::data<double>(0, ade::Shape({n_output}), "bias"));
			weight_bias_.push_back({weight, bias});
		}
	}

	virtual ~FCLayer (void) {}

	FCLayer (const FCLayer& other, std::string label_prefix = "copied_") :
		iLayer(label_prefix + other.label_)
	{
		copy_helper(other);
	}

	FCLayer& operator = (const FCLayer& other)
	{
		if (this != &other)
		{
			label_ = "copied_" + other.label_;
			copy_helper(other);
		}
		return *this;
	}

	FCLayer (FCLayer&& other) : iLayer(other.label_)
	{
		weight_bias_ = std::move(other.weight_bias_);
	}

	FCLayer& operator = (FCLayer&& other)
	{
		if (this != &other)
		{
			label_ = std::move(other.label_);
			weight_bias_ = std::move(other.weight_bias_);
		}
		return *this;
	}


	ade::Tensorptr operator () (age::TensT inputs)
	{
		size_t n = inputs.size();
		if (n != weight_bias_.size())
		{
			err::fatalf("number of inputs must be exactly %d", n);
		}
		age::TensT args;
		for (size_t i = 0; i < n; ++i)
		{
			ade::DimT cdim = inputs[i]->shape().at(1);
			args.push_back(age::matmul(inputs[i],
				ade::Tensorptr(weight_bias_[i].first)));
			args.push_back(age::extend(
				ade::Tensorptr(weight_bias_[i].second), 1, {cdim}));
		}
		return age::sum(args);
	}

	std::vector<llo::VarptrT> get_variables (void) const override
	{
		std::vector<llo::VarptrT> out;
		for (const WbPairT& wb : weight_bias_)
		{
			out.push_back(wb.first);
			out.push_back(wb.second);
		}
		return out;
	}

protected:
	using WbPairT = std::pair<llo::VarptrT,llo::VarptrT>;

	std::vector<WbPairT> weight_bias_;

private:
	void copy_helper (const FCLayer& other)
	{
		weight_bias_.clear();
		for (const WbPairT& opair : other.weight_bias_)
		{
			llo::VarptrT ow(new llo::Variable(*opair.first));
			llo::VarptrT ob(new llo::Variable(*opair.second));
			weight_bias_.push_back({ow, ob});
		}
	}
};

#endif // LAYR_FC_LAYER_HPP
