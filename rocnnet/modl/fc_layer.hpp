#include "pbm/load.hpp"

#include "llo/operator.hpp"

#ifndef MODL_FC_LAYER_HPP
#define MODL_FC_LAYER_HPP

struct FCLayer
{
	FCLayer (std::vector<uint8_t> n_inputs, uint8_t n_output, std::string label) :
		label_("fc_" + label)
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
				llo::get_variable(data, shape, err::sprintf("weight_%d", i)));
			llo::VarptrT bias(
				llo::data<double>(0, ade::Shape({n_output}), err::sprintf("bias_%d", i)));
			weight_bias_.push_back({weight, bias});
		}
	}

	FCLayer (const FCLayer& other) :
		label_(other.label_)
	{
		copy_helper(other);
	}

	FCLayer& operator = (const FCLayer& other)
	{
		if (this != &other)
		{
			other.label_;
			copy_helper(other);
		}
		return *this;
	}

	FCLayer (FCLayer&& other) = default;

	FCLayer& operator = (FCLayer&& other) = default;


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

	std::vector<llo::VarptrT> get_variables (void) const
	{
		std::vector<llo::VarptrT> out;
		for (const WbPairT& wb : weight_bias_)
		{
			out.push_back(wb.first);
			out.push_back(wb.second);
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

	void parse_from (pbm::LoadVecsT labels)
	{
		std::unordered_map<std::string,ade::Tensorptr> relevant;
		for (pbm::LoadTensT& pairs : labels)
		{
			if (pairs.second.size() == 2 &&
				label_ == pairs.second.front())
			{
				relevant.emplace(pairs.second.back(), pairs.first);
			}
		}
		for (size_t i = 0, n = weight_bias_.size(); i < n; ++i)
		{
			std::string weight_label = err::sprintf("weight_%d", i);
			std::string bias_label = err::sprintf("bias_%d", i);
			auto wit = relevant.find(weight_label);
			if (relevant.end() == wit)
			{
				err::warn(weight_label + " not found in protobuf");
			}
			else
			{
				wit->second;
	            weight_bias_[i].first;
			}
			auto bit = relevant.find(bias_label);
			if (relevant.end() == wit)
			{
				err::warn(bias_label + " not found in protobuf");
			}
			else
			{
				bit->second;
            	weight_bias_[i].second;
			}
		}
	}

private:
	using WbPairT = std::pair<llo::VarptrT,llo::VarptrT>;

	std::vector<WbPairT> weight_bias_;

	std::string label_;

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

#endif // MODL_FC_LAYER_HPP
