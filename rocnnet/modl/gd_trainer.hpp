#include "llo/eval.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#include "rocnnet/modl/mlp.hpp"

// GDTrainer does not own anything
struct GDTrainer
{
	GDTrainer (MLP& brain, ApproxFuncT update,
		uint8_t batch_size, std::string label) :
		label_(label), brain_(&brain), batch_size_(batch_size),
		train_in_(llo::data<double>(0,
			ade::Shape({brain.get_ninput(), batch_size}), "train_in")),
		train_out_(brain(ade::Tensorptr(train_in_))),
		expected_out_(llo::data<double>(0,
			ade::Shape({brain.get_noutput(), batch_size}), "expected_out")),
		// todo: move error out of initializer list to avoid confusing order of init
		error_(age::pow(age::sub(ade::Tensorptr(expected_out_), train_out_),
			age::data(2, expected_out_->shape())))
	{
		updates_ = update(error_, brain.get_variables());
	}

	void train (std::vector<double>& train_in,
		std::vector<double>& expected_out)
	{
		size_t insize = brain_->get_ninput();
		size_t outsize = brain_->get_noutput();
		if (train_in.size() != insize * batch_size_)
		{
			err::fatalf("training vector size (%d) does not match input size "
				"(%d) * batchsize (%d)", train_in.size(), insize, batch_size_);
		}
		if (expected_out.size() != outsize * batch_size_)
		{
			err::fatalf("expected output size (%d) does not match output size "
				"(%d) * batchsize (%d)", expected_out.size(), outsize, batch_size_);
		}
		*train_in_ = train_in;
		*expected_out_ = expected_out;

		std::unordered_map<llo::Variable*,llo::GenericData> data;
		for (auto& varpair : updates_)
		{
			data[varpair.first] = llo::eval(varpair.second, age::DOUBLE);
		}
		for (auto& datapair : data)
		{
			*datapair.first = llo::GenericRef(datapair.second);
		}
	}

	std::string label_;
	MLP* brain_ = nullptr; // do not own this
	uint8_t batch_size_;
	llo::VarptrT train_in_;
	ade::Tensorptr train_out_;
	llo::VarptrT expected_out_;
	ade::Tensorptr error_;

	DeltasT updates_;
};
