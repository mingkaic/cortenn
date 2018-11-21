#include <functional>
#include <memory>

#include "rocnnet/layr/fc_layer.hpp"

using HiddenFunc = std::function<ade::Tensorptr(ade::Tensorptr)>;

struct LayerInfo
{
	size_t n_out_;
	HiddenFunc hidden_;
};

struct HiddenLayer
{
	ade::Tensorptr operator () (ade::Tensorptr& input)
	{
		auto hypothesis = layer_({input});
		return hidden_(hypothesis);
	}

	FCLayer layer_;
	HiddenFunc hidden_;
};

struct MLP
{
	MLP (uint8_t n_input, std::vector<LayerInfo> layers, std::string label) :
		label_(label)
	{
		for (size_t i = 0, n = layers.size(); i < n; ++i)
		{
			size_t n_output = layers[i].n_out_;
			layers_.push_back(HiddenLayer{
				FCLayer(std::vector<uint8_t>{n_input}, n_output,
					err::sprintf("hidden_%d", i)),
				layers[i].hidden_
			});
			n_input = n_output;
		}
	}

	MLP (std::vector<HiddenLayer> layers, std::string label) :
		label_(label), layers_(layers) {}

	MLP (const MLP& other, std::string label_prefix = "copied_")
	{
		copy_helper(other, label_prefix);
	}

	MLP& operator = (const MLP& other)
	{
		if (this != &other)
		{
			copy_helper(other, "copied_");
		}
		return *this;
	}

	MLP (MLP&& other) = default;

	MLP& operator = (MLP&& other) = default;


	ade::Tensorptr operator () (ade::Tensorptr input)
	{
		ade::Tensorptr out = input;
		for (HiddenLayer& layer : layers_)
		{
			out = layer(out);
		}
		return out;
	}

	std::vector<llo::VarptrT> get_variables (void) const
	{
		std::vector<llo::VarptrT> out;
		for (const HiddenLayer& layer : layers_)
		{
			auto temp = layer.layer_.get_variables();
			out.insert(out.end(), temp.begin(), temp.end());
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

private:
	void copy_helper (const MLP& other, std::string prefix)
	{
		label_ = prefix + other.label_;
		layers_.clear();
		for (const HiddenLayer& olayer : other.layers_)
		{
			layers_.push_back(HiddenLayer{
				FCLayer(olayer.layer_, prefix),
				olayer.hidden_
			});
		}
	}

	std::string label_;

	std::vector<HiddenLayer> layers_;
};
