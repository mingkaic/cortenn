#include "llo/eval.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#include "rocnnet/modl/mlp.hpp"

#ifndef MODL_GD_TRAINER_HPP
#define MODL_GD_TRAINER_HPP

// GDTrainer does not own anything
struct GDTrainer
{
	GDTrainer (modl::MLptrT brain, eqns::ApproxFuncT update,
		uint8_t batch_size, std::string label) :
		label_(label), brain_(brain), batch_size_(batch_size),
		train_in_(llo::data<double>(0,
			ade::Shape({brain->get_ninput(), batch_size}), "train_in"))
	{
		train_out_ = (*brain_)(ade::TensptrT(train_in_));
		expected_out_ = llo::data<double>(0,
			ade::Shape({brain_->get_noutput(), batch_size}), "expected_out");
		error_ = age::pow(age::sub(ade::TensptrT(expected_out_), train_out_),
			ade::TensptrT(age::data(2, expected_out_->shape())));

		pbm::PathedMapT vmap = brain_->list_bases();
		eqns::VariablesT vars;
		for (auto vpair : vmap)
		{
			if (llo::VarptrT var = std::dynamic_pointer_cast<
				llo::Variable>(vpair.first))
			{
				vars.push_back(var);
			}
		}
		updates_ = update(error_, vars);
	}

	void train (std::vector<double>& train_in,
		std::vector<double>& expected_out)
	{
		size_t insize = brain_->get_ninput();
		size_t outsize = brain_->get_noutput();
		if (train_in.size() != insize * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match input size "
				"(%d) * batchsize (%d)", train_in.size(), insize, batch_size_);
		}
		if (expected_out.size() != outsize * batch_size_)
		{
			logs::fatalf("expected output size (%d) does not match output size "
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
	modl::MLptrT brain_;
	uint8_t batch_size_;
	llo::VarptrT train_in_;
	ade::TensptrT train_out_;
	llo::VarptrT expected_out_;
	ade::TensptrT error_;

	eqns::DeltasT updates_;
};

#endif // MODL_GD_TRAINER_HPP
