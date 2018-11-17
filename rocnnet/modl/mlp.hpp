#include <functional>
#include <memory>

#include "rocnnet/layr/fc_layer.hpp"

using HiddenFunc = std::function<ade::Tensorptr(ade::Tensorptr)>;

struct LayerInfo
{
	size_t n_out_;
	HiddenFunc hidden_;
};

struct MLP
{
	MLP (uint8_t n_input, std::vector<LayerInfo> layers, std::string label) :
		label_(label), n_input_(n_input)
	{
		for (size_t i = 0, n = layers.size(); i < n; ++i)
		{
			n_output_ = layers[i].n_out_;
			std::stringstream ss;
			ss << label << ":hidden_" << i;
			layers_.push_back(Layer{
				std::make_unique<FCLayer>(
					std::vector<uint8_t>{n_input}, n_output_, ss.str()),
				layers[i].hidden_
			});
			n_input = n_output_;
		}
	}

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

	MLP (MLP&& other)
	{
		move_helper(std::move(other));
	}

	MLP& operator = (MLP&& other)
	{
		if (this != &other)
		{
			label_ = other.label_;
			move_helper(std::move(other));
		}
		return *this;
	}


	ade::Tensorptr operator () (ade::Tensorptr input)
	{
		ade::Tensorptr out = input;
		for (Layer& layer : layers_)
		{
			out = layer(out);
		}
		return out;
	}

	std::vector<llo::VarptrT> get_variables (void) const
	{
		std::vector<llo::VarptrT> out;
		for (const Layer& layer : layers_)
		{
			auto temp = layer.layer_->get_variables();
			out.insert(out.end(), temp.begin(), temp.end());
		}
		return out;
	}

	uint8_t get_ninput (void) const
	{
		return n_input_;
	}

	uint8_t get_noutput (void) const
	{
		return n_output_;
	}

private:
	void copy_helper (const MLP& other, std::string prefix)
	{
		label_ = prefix + other.label_;
		n_input_ = other.n_input_;
		n_output_ = other.n_output_;
		layers_.clear();
		for (const Layer& olayer : other.layers_)
		{
			layers_.push_back(Layer{
				std::make_unique<FCLayer>(*(olayer.layer_.get()), prefix),
				olayer.hidden_
			});
		}
	}

	void move_helper (MLP&& other)
	{
		label_ = std::move(other.label_);
		n_input_ = std::move(other.n_input_);
		n_output_ = std::move(other.n_output_);
		layers_ = std::move(other.layers_);
	}

	std::string label_;
	uint8_t n_input_;
	uint8_t n_output_;

	struct Layer
	{
		ade::Tensorptr operator () (ade::Tensorptr& input)
		{
			auto hypothesis = (*layer_)({input});
			return hidden_(hypothesis);
		}

		std::unique_ptr<FCLayer> layer_;
		HiddenFunc hidden_;
	};

	std::vector<Layer> layers_;
};

