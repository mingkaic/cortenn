#include <functional>
#include <memory>

#include "rocnnet/modl/fc_layer.hpp"

#ifndef MODL_MLP_HPP
#define MODL_MLP_HPP

using HiddenFunc = std::function<ade::TensptrT(ade::TensptrT)>;

struct LayerInfo
{
	size_t n_out_;
	HiddenFunc hidden_;
};

struct MLP final
{
	MLP (uint8_t n_input, std::vector<LayerInfo> layers, std::string label) :
		label_("mlp_" + label)
	{
		for (size_t i = 0, n = layers.size(); i < n; ++i)
		{
			size_t n_output = layers[i].n_out_;
			layers_.push_back(HiddenLayer{
				FCLayer(std::vector<uint8_t>{n_input}, n_output,
					fmts::sprintf("hidden_%d", i)),
				layers[i].hidden_
			});
			n_input = n_output;
		}
	}

	MLP (const MLP& other)
	{
		copy_helper(other);
	}

	MLP& operator = (const MLP& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	MLP (MLP&& other) = default;

	MLP& operator = (MLP&& other) = default;


	ade::TensptrT operator () (ade::TensptrT input)
	{
		ade::TensptrT out = input;
		for (HiddenLayer& layer : layers_)
		{
			out = layer(out);
		}
		return out;
	}

	std::vector<LabelVar> get_variables (void) const
	{
		std::vector<LabelVar> out;
		for (const HiddenLayer& layer : layers_)
		{
			auto temp = layer.layer_.get_variables();
			out.insert(out.end(), temp.begin(), temp.end());
		}
		for (LabelVar& lv : out)
		{
			lv.labels_.push_front(label_);
		}
		return out;
	}

	uint8_t get_ninput (void) const
	{
		return layers_.front().layer_.get_ninput();
	}

	uint8_t get_noutput (void) const
	{
		return layers_.back().layer_.get_noutput();
	}

	void parse_from (pbm::LabelledsT labels)
	{
		pbm::LabelledsT relevant;
		std::copy_if(labels.begin(), labels.end(), std::back_inserter(relevant),
			[&](pbm::LabelledTensT& pairs)
			{
				return pairs.second.size() > 0 &&
					this->label_ == pairs.second.front();
			});
		for (pbm::LabelledTensT& pairs : relevant)
		{
			pairs.second.pop_front();
		}
		for (HiddenLayer& olayer : layers_)
		{
			olayer.layer_.parse_from(relevant);
		}
	}

private:
	void copy_helper (const MLP& other)
	{
		label_ = other.label_;
		layers_.clear();
		for (const HiddenLayer& olayer : other.layers_)
		{
			layers_.push_back(HiddenLayer{
				FCLayer(olayer.layer_), olayer.hidden_
			});
		}
	}

	struct HiddenLayer
	{
		ade::TensptrT operator () (ade::TensptrT& input)
		{
			auto hypothesis = layer_({input});
			return hidden_(hypothesis);
		}

		FCLayer layer_;
		HiddenFunc hidden_;
	};

	std::string label_;

	std::vector<HiddenLayer> layers_;
};

#endif // MODL_MLP_HPP
