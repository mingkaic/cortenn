#include <random>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iostream>
#include <fstream>

#include "dbg/ade.hpp"

#include "pbm/save.hpp"

#include "llo/source.hpp"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/modl/gd_trainer.hpp"

#include "rocnnet/demo/options.hpp"

Options options;

static std::vector<double> batch_generate (size_t n, size_t batchsize)
{
	size_t total = n * batchsize;

	// Specify the engine and distribution.
	std::mt19937 mersenne_engine(llo::get_engine());
	std::uniform_real_distribution<double> dist(0, 1);

	auto gen = std::bind(dist, mersenne_engine);
	std::vector<double> vec(total);
	std::generate(std::begin(vec), std::end(vec), gen);
	return vec;
}

static std::vector<double> avgevry2 (std::vector<double>& in)
{
	std::vector<double> out;
	for (size_t i = 0, n = in.size()/2; i < n; i++)
	{
		double val = (in.at(2*i) + in.at(2*i+1)) / 2;
		out.push_back(val);
	}
	return out;
}

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	std::string savepath;
	std::string loadpath;
	size_t n_train;
	size_t n_test;

	options.desc_.add_options()
		("load", opt::value<std::string>(&loadpath)->default_value("rocnnet/pretrained/gdmodel.pbx"),
			"filename to load pretrained model")
		("save", opt::value<std::string>(&savepath)->default_value(""),
			"filename to save model")
		("n_train", opt::value<size_t>(&n_train)->default_value(3000),
			"number of times to train")
		("n_test", opt::value<size_t>(&n_test)->default_value(500),
			"number of times to test");

	int exit_status = 0;
	std::clock_t start;
	double duration;

	if (false == options.parse(argc, argv))
	{
		return 1;
	}

	if (options.seed_)
	{
		size_t seed = options.seedval_;
		std::cout << "seeding " << seed << '\n';
		llo::get_engine().seed(seed);
	}

	uint8_t n_in = 10;
	uint8_t n_out = n_in / 2;

	std::vector<LayerInfo> hiddens = {
		// use same sigmoid in static memory once models copy is established
		LayerInfo{9, sigmoid},
		LayerInfo{n_out, sigmoid}
	};
	MLP brain(n_in, hiddens, "brain");
	MLP untrained_brain(brain);
	MLP pretrained_brain(brain);
	std::ifstream loadstr(loadpath);
	if (loadstr.is_open())
	{
		tenncor::Graph graph;
		graph.ParseFromIstream(&loadstr);
		pbm::GraphInfo info;
		pbm::load_graph(info, graph, llo::deserialize);
		pbm::LabelledsT vars = info.labelled_;
		pretrained_brain.parse_from(vars);
		loadstr.close();
	}

	uint8_t n_batch = 3;
	size_t show_every_n = 500;
	ApproxFuncT approx = [](ade::TensptrT& root, VariablesT leaves)
	{
		return sgd(root, leaves, 0.9); // learning rate = 0.9
	};
	GDTrainer trainer(brain, approx, n_batch, "gdn");

	PrettyEquation peq;
	size_t i = 0;
	for (auto deltas : trainer.updates_)
	{
		peq.labels_[deltas.first] = err::sprintf("var%d", i);
		++i;
	}
	peq.labels_[trainer.expected_out_.get()] = "expected_out";
	peq.labels_[trainer.error_.get()] = "error";
	peq.labels_[trainer.train_in_.get()] = "train_in";
	peq.print(std::cout, trainer.error_);
	std::cout << '\n';
#if 0
	for (auto deltas : trainer.updates_)
	{
		peq.print(std::cout, deltas.second);
		std::cout << '\n';
	}
#endif

	std::vector<double> batch = batch_generate(n_in, n_batch);
	std::vector<double> batch_out = avgevry2(batch);
	*trainer.train_in_ = batch;
	*trainer.expected_out_ = batch_out;

	// train mlp to output input
	start = std::clock();
	for (size_t i = 0; i < n_train; i++)
	{
		if (i % show_every_n == show_every_n-1)
		{
			llo::GenericData trained_derr = llo::eval(trainer.error_, age::DOUBLE);
			double* trained_err_res = (double*) trained_derr.data_.get();
			std::cout << "training " << i+1 << '\n';
			std::cout << "trained error: " << err::to_string(trained_err_res, trained_err_res + n_out) << '\n';
		}
		std::vector<double> batch = batch_generate(n_in, n_batch);
		std::vector<double> batch_out = avgevry2(batch);
		trainer.train(batch, batch_out);
	}
	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	std::cout << "training time: " << duration << " seconds" << '\n';

	// exit code:
	//	0 = fine
	//	1 = training error rate is wrong
	double untrained_err = 0;
	double trained_err = 0;
	double pretrained_err = 0;

	llo::VarptrT testin = llo::get_variable(
		std::vector<double>(n_in), ade::Shape({n_in}), "testin");
	auto untrained_out = untrained_brain(testin);
	auto trained_out = brain(testin);
	auto pretrained_out = pretrained_brain(testin);
	for (size_t i = 0; i < n_test; i++)
	{
		if (i % show_every_n == show_every_n-1)
		{
			std::cout << "testing " << i+1 << '\n';
		}
		std::vector<double> batch = batch_generate(n_in, 1);
		std::vector<double> batch_out = avgevry2(batch);
		*testin = batch;

		llo::GenericData untrained_data = llo::eval(untrained_out, age::DOUBLE);
		llo::GenericData trained_data = llo::eval(trained_out, age::DOUBLE);
		llo::GenericData pretrained_data = llo::eval(pretrained_out, age::DOUBLE);

		double* untrained_res = (double*) untrained_data.data_.get();
		double* trained_res = (double*) trained_data.data_.get();
		double* pretrained_res = (double*) pretrained_data.data_.get();

		double untrained_avgerr = 0;
		double trained_avgerr = 0;
		double pretrained_avgerr = 0;
		for (size_t i = 0; i < n_out; i++)
		{
			untrained_avgerr += std::abs(untrained_res[i] - batch_out[i]);
			trained_avgerr += std::abs(trained_res[i] - batch_out[i]);
			pretrained_avgerr += std::abs(pretrained_res[i] - batch_out[i]);
		}
		untrained_err += untrained_avgerr / n_out;
		trained_err += trained_avgerr / n_out;
		pretrained_err += pretrained_avgerr / n_out;
	}
	untrained_err /= (double) n_test;
	trained_err /= (double) n_test;
	pretrained_err /= (double) n_test;
	std::cout << "untrained mlp error rate: " << untrained_err * 100 << "%\n";
	std::cout << "trained mlp error rate: " << trained_err * 100 << "%\n";
	std::cout << "pretrained mlp error rate: " << pretrained_err * 100 << "%\n";

	// try to save
	if (exit_status == 0)
	{
		std::ofstream savestr(savepath);
		if (savestr.is_open())
		{
			pbm::GraphSaver saver(llo::serialize);
			trained_out->accept(saver);

			std::vector<LabelVar> vars = brain.get_variables();
			pbm::TensLabelT labels;
			for (LabelVar& var : vars)
			{
				labels[var.var_.get()] = var.labels_;
			}

			tenncor::Graph graph;
			saver.save(graph, labels);
			graph.SerializeToOstream(&savestr);
			savestr.close();
		}
	}

	google::protobuf::ShutdownProtobufLibrary();

	return exit_status;
}
