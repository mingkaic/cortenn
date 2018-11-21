#include <functional>
#include <memory>

#include "rocnnet/modl/fc_layer.hpp"

using HiddenFunc = std::function<ade::Tensorptr(ade::Tensorptr)>;

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
					err::sprintf("hidden_%d", i)),
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
		ade::Tensorptr operator () (ade::Tensorptr& input)
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
