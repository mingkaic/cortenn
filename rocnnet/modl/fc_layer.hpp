#include "pbm/load.hpp"

#include "llo/operator.hpp"

#ifndef MODL_FC_LAYER_HPP
#define MODL_FC_LAYER_HPP

const std::string fc_prefix = "fc_";
const std::string weight_fmt = "weight_%d";
const std::string bias_fmt = "bias_%d";

struct LabelVar
{
	llo::VarptrT var_;
	pbm::StringsT labels_;
};

struct FCLayer
{
	FCLayer (std::vector<uint8_t> n_inputs, uint8_t n_output, std::string label) :
		label_(fc_prefix + label)
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

			weight_bias_.push_back({
				llo::VarptrT(llo::get_variable(data, shape,
					err::sprintf(weight_fmt, i))),
				llo::VarptrT(llo::data<double>(0, ade::Shape({n_output}),
					err::sprintf(bias_fmt, i)))});
		}
	}

	FCLayer (const FCLayer& other)
	{
		copy_helper(other);
	}

	FCLayer& operator = (const FCLayer& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	FCLayer (FCLayer&& other) = default;

	FCLayer& operator = (FCLayer&& other) = default;


	ade::TensptrT operator () (age::TensT inputs)
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
				ade::TensptrT(weight_bias_[i].first)));
			args.push_back(age::extend(
				ade::TensptrT(weight_bias_[i].second), 1, {cdim}));
		}
		return age::sum(args);
	}

	std::vector<LabelVar> get_variables (void) const
	{
		std::vector<LabelVar> out;
		for (size_t i = 0, n = weight_bias_.size(); i < n; ++i)
		{
			std::string weight_label = err::sprintf(weight_fmt, i);
			std::string bias_label = err::sprintf(bias_fmt, i);
			out.push_back({
				weight_bias_[i].first,
				{label_, weight_label}
			});
			out.push_back({
				weight_bias_[i].second,
				{label_, bias_label}
			});
		}
		return out;
	}

	uint8_t get_ninput (void) const
	{
		return weight_bias_[0].first->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_bias_[0].first->shape().at(0);
	}

	void parse_from (pbm::LabelledsT labels)
	{
		std::unordered_map<std::string,ade::TensptrT> relevant;
		for (pbm::LabelledTensT& pairs : labels)
		{
			if (label_ == pairs.second.front())
			{
				relevant.emplace(pairs.second.back(), pairs.first);
			}
		}
		for (size_t i = 0, n = weight_bias_.size(); i < n; ++i)
		{
			std::string weight_label = err::sprintf(weight_fmt, i);
			std::string bias_label = err::sprintf(bias_fmt, i);
			auto wit = relevant.find(weight_label);
			if (relevant.end() == wit)
			{
				err::warn(weight_label + " not found in protobuf");
			}
			else
			{
				llo::GenericData wdata = llo::eval(wit->second, age::DOUBLE);
				double* wptr = (double*) wdata.data_.get();
	            *(weight_bias_[i].first) = std::vector<double>(wptr, wptr + wdata.shape_.n_elems());
			}
			auto bit = relevant.find(bias_label);
			if (relevant.end() == wit)
			{
				err::warn(bias_label + " not found in protobuf");
			}
			else
			{
				llo::GenericData bdata = llo::eval(bit->second, age::DOUBLE);
				double* bptr = (double*) bdata.data_.get();
            	*(weight_bias_[i].second) = std::vector<double>(bptr, bptr + bdata.shape_.n_elems());
			}
		}
	}

private:
	using WbPairT = std::pair<llo::VarptrT,llo::VarptrT>;

	std::vector<WbPairT> weight_bias_;

	std::string label_;

	void copy_helper (const FCLayer& other)
	{
		label_ = other.label_;
		weight_bias_.clear();
		for (const WbPairT& opair : other.weight_bias_)
		{
			llo::VarptrT ow(new llo::Variable(*opair.first));
			llo::VarptrT ob(new llo::Variable(*opair.second));
			weight_bias_.push_back({ow, ob});
		}
	}
};

#endif // MODL_FC_LAYER_HPP
