//
// Created by Mingkai Chen on 2017-09-21.
//

#include <random>

#include "rocnnet/models/db_trainer.hpp"
#include "rocnnet/data/mnist_data.hpp"

#include "rocnnet/demo/options.hpp"

Options options;

struct TestParams
{
	double pretrain_lr = 0.01;
	double training_lr = 0.1;
	size_t n_batch = 20;
	std::vector<size_t> hiddens = { 1000, 1000, 1000 };

	size_t n_cont_div;
	size_t pretrain_epochs;
	size_t training_epochs;

	bool pretrain;
	bool train;

	std::string pretrain_path;
	std::string savepath;
	std::string loadpath;
};

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

static void pretrain (DBTrainer& model, size_t n_input,
	std::vector<double> data, TestParams params, std::string test_name)
{
	std::string pretrain_path = params.pretrain_path;
	size_t n_data = data.size() / n_input;
	size_t n_train_batches = n_data / params.n_batch;
	llo::VarptrT pretrain_in = llo::get_variable(
		std::vector<double>(n_input * params.n_batch),
		ade::Shape({n_input, params.n_batch}), "pretrain_in");
	double inbatch = params.n_batch * n_input;

	std::cout << "... getting the pretraining functions" << '\n';
	PretrainsT pretrainers = model.pretraining_functions(
		pretrain_in, params.pretrain_lr , params.n_cont_div);

	std::cout << "... pre-training the model" << '\n';
	auto it = data.begin();
	for (size_t pidx = 0; pidx < pretrainers.size(); pidx++)
	{
		DeltasNCostT& ptit = pretrainers[pidx];
		DeltasT trainer = ptit.first;
		ade::TensptrT cost = ptit.second;
		for (size_t e = 0; e < params.pretrain_epochs; e++)
		{
			double mean_cost = 0;
			for (size_t i = 0; i < n_train_batches; i++)
			{
				std::vector<double> batch(it + i * inbatch, it + (i + 1) * inbatch);
				pretrain_in = batch;
				trainer(true);
				std::cout << "layer " << pidx << " epoch " << e << " completed batch " << i << '\n';

				llo::GenericData gcost = llo::eval(cost, llo::DOUBLE);
				mean_cost += *((double*) gcost.data_.get());
			}
			std::cout << "pre-trained layer " << pidx << " epoch "
				<< e << " has mean cost " << mean_cost / n_train_batches << '\n';

			std::ofstream savestr(pretrain_path);
			if (savestr.is_open())
			{
				pbm::GraphSaver saver(llo::serialize);

				std::vector<LabelVar> vars = model.get_variables();
				pbm::TensLabelT labels;
				for (LabelVar& var : vars)
				{
					labels[var.var_.get()] = var.labels_;
					var.var_->accept(saver);
				}

				tenncor::Graph graph;
				saver.save(graph, labels);
				graph.SerializeToOstream(&savestr);
				savestr.close();
			}
		}
	}
}

static void finetune (DBTrainer& model, ImgPtrT train,
	ImgPtrT valid, ImgPtrT test, size_t n_input,
	size_t n_output, TestParams params)
{
	size_t n_train_batches = train->shape_.first;

	std::vector<double> training_data_x(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> training_data_y(train->data_y_.begin(), train->data_y_.end());

	std::cout << "... getting the finetuning functions" << '\n';
	nnet::placeholder<double> finetune_in(std::vector<size_t>{n_input, params.n_batch}, "finetune_in");
	nnet::placeholder<double> finetune_out(std::vector<size_t>{n_output, params.n_batch}, "finetune_out");
	rocnnet::update_cost_t tuner = model.build_finetune_functions(finetune_in, finetune_out, params.training_lr);
	nnet::variable_updater<double> train_update = tuner.first;
	nnet::varptr<double> train_cost = tuner.second;
	nnet::varptr<double> train_loss = nnet::reduce_mean(train_cost);

	std::cout << "... finetuning the model" << '\n';
	size_t patience = 4 * n_train_batches;
	size_t patience_increase = 2;
	size_t validation_frequency = std::min(n_train_batches, patience / 2);
	double improvement_threshold = 0.995;
	double best_validation_loss = std::numeric_limits<double>::infinity();
	double test_score = 0;
	bool keep_looping = true;
	size_t best_iter = 0;

	size_t inbatch = params.n_batch * n_input;
	size_t outbatch = params.n_batch * n_output;
	auto xit = training_data_x.begin();
	auto yit = training_data_y.begin();

	std::vector<double> valid_data_x(valid->data_x_.begin(), valid->data_x_.end());
	std::vector<double> valid_data_y(valid->data_y_.begin(), valid->data_y_.end());
	std::vector<double> test_data_x(test->data_x_.begin(), test->data_x_.end());
	std::vector<double> test_data_y(test->data_y_.begin(), test->data_y_.end());
	auto valid_xit = valid_data_x.begin();
	auto valid_yit = valid_data_y.begin();
	auto test_xit = test_data_x.begin();
	auto test_yit = test_data_y.begin();

	for (size_t epoch = 0; epoch < params.training_epochs && keep_looping; epoch++)
	{
		for (size_t mb_idx = 0; mb_idx < n_train_batches; mb_idx++)
		{
			std::vector<double> xbatch(xit + mb_idx * inbatch, xit + (mb_idx + 1) * inbatch);
			std::vector<double> ybatch(yit + mb_idx * outbatch, yit + (mb_idx + 1) * outbatch);
			finetune_in = xbatch;
			finetune_out = ybatch;
			train_update(true);

			size_t iter = (epoch - 1) * n_train_batches + mb_idx;

			if (((iter + 1) % validation_frequency) == 0)
			{
				std::vector<double> xbatch_valid(valid_xit + mb_idx * inbatch, valid_xit + (mb_idx + 1) * inbatch);
				std::vector<double> ybatch_valid(valid_yit + mb_idx * outbatch, valid_yit + (mb_idx + 1) * outbatch);
				finetune_in = xbatch_valid;
				finetune_out = ybatch_valid;

				double validate_loss = nnet::expose<double>(train_loss)[0];
				std::cout << "epoch " << epoch << ", minibatch "
					<< mb_idx + 1 << "/" << n_train_batches << ", validation error "
					<< validate_loss << '\n';

				if (validate_loss < best_validation_loss)
				{
					// improve patience if loss improvement is good enough
					if (validate_loss < best_validation_loss * improvement_threshold)
					{
						patience = std::max(patience, iter * patience_increase);
					}

					best_validation_loss = validate_loss;
					best_iter = iter;

					std::vector<double> xbatch_test(test_xit + mb_idx * inbatch, test_xit + (mb_idx + 1) * inbatch);
					std::vector<double> ybatch_test(test_yit + mb_idx * outbatch, test_yit + (mb_idx + 1) * outbatch);
					finetune_in = xbatch_test;
					finetune_out = ybatch_test;

					double test_loss = nnet::expose<double>(train_loss)[0];
					std::cout << "\tepoch " << epoch << ", minibatch "
						<< mb_idx + 1 << "/" << n_train_batches << ", test error of best model "
						<< test_loss * 100.0 << '\n';
				}
			}

			if (patience <= iter)
			{
				keep_looping = false;
				break;
			}
		}
	}

	std::cout << "Optimization complete with best validation score of "
		<< best_validation_loss * 100.0 << ", obtained at iteration "
		<< best_iter + 1 << ", with test performance "
		<< test_score * 100.0 << '\n';
}

static void mnist_test (ImgPtrT train, ImgPtrT valid, ImgPtrT test, TestParams params)
{
	std::string serialpath = params.savepath;

	std::vector<double> training_data_x(train->data_x_.begin(), train->data_x_.end());
	std::vector<double> training_data_y(train->data_y_.begin(), train->data_y_.end());
	std::vector<double> valid_data_x(valid->data_x_.begin(), valid->data_x_.end());
	std::vector<double> valid_data_y(valid->data_y_.begin(), valid->data_y_.end());
	std::vector<double> test_data_x(test->data_x_.begin(), test->data_x_.end());
	std::vector<double> test_data_y(test->data_y_.begin(), test->data_y_.end());

	size_t n_input = train->shape_.first;
	size_t n_output = 10;
	params.hiddens.push_back(n_output);

	DBTrainer model(n_input, params.hiddens, "dbn_mnist_learner");

	if (params.pretrain)
	{
		model.initialize();
		pretrain(model, n_input, training_data_x, params, "mnist");

		model.save(serialpath, "dbn_mnist_pretrain");
	}
	else
	{
		model.initialize(serialpath, "dbn_mnist_pretrain");
	}

	finetune(model, train, valid, test, n_input, n_output, params);

	model.save(serialpath, "dbn_mnist");
}

std::vector<double> simple_op (std::vector<double> input)
{
	std::vector<double> output;
	for (size_t i = 0, n = input.size() / 2; i < n; ++i)
	{
		output.push_back((input[i] + input[n + i]) / 2);
	}
	return output;
}

void simpler_test (size_t n_train_sample, size_t n_test_sample, size_t n_in, TestParams params)
{
	params.n_batch = std::min(params.n_batch, n_train_sample);
	std::string serialpath = params.savepath;
	params.hiddens = { n_in, n_in, n_in / 2 };
	DBTrainer model(n_in, params.hiddens, "dbn_simple_learner");

	// generate test sample
	std::vector<double> train_samples = batch_generate(n_train_sample, n_in);
	std::vector<double> test_samples = batch_generate(n_test_sample, n_in);
	std::vector<double> train_out = simple_op(train_samples);
	std::vector<double> test_out = simple_op(test_samples);

	if (params.train)
	{
		// pretrain
		if (params.pretrain)
		{
			model.initialize();
			pretrain(model, n_in, train_samples, params, "demo");

			model.save(serialpath, "dbn_demo_pretrain");
		}
		else
		{
			model.initialize(serialpath, "dbn_demo_pretrain");
		}

		// finetune
		double inbatch = params.n_batch * n_in;
		double outbatch = inbatch / 2;
		nnet::placeholder<double> finetune_in(std::vector<size_t>{n_in, params.n_batch}, "finetune_in");
		nnet::placeholder<double> finetune_out(std::vector<size_t>{n_in / 2, params.n_batch}, "finetune_out");
		rocnnet::update_cost_t tuner = model.build_finetune_functions(finetune_in, finetune_out, params.training_lr);
		nnet::variable_updater<double> train_update = tuner.first;
		size_t n_train_batches = n_train_sample / params.n_batch;

		auto xit = train_samples.begin();
		auto yit = train_out.begin();

		for (size_t epoch = 0; epoch < params.training_epochs; epoch++)
		{
			for (size_t mb_idx = 0; mb_idx < n_train_batches; mb_idx++)
			{
				std::vector<double> xbatch(xit + mb_idx * inbatch, xit + (mb_idx + 1) * inbatch);
				std::vector<double> ybatch(yit + mb_idx * outbatch, yit + (mb_idx + 1) * outbatch);
				finetune_in = xbatch;
				finetune_out = ybatch;
				train_update(true);
				std::cout << "epoch " << epoch << " fine tuning index " << mb_idx << '\n';
			}
		}

		model.save(serialpath, "dbn_demo");
	}
	else
	{
		model.initialize(serialpath, "dbn_demo");
	}

	// test
	nnet::placeholder<double> test_in(std::vector<size_t>{n_in}, "test_in");
	nnet::placeholder<double> expect_out(std::vector<size_t>{n_in / 2}, "expect_out");
	nnet::varptr<double> test_res = model.prop_up(nnet::varptr<double>(&test_in));
	nnet::varptr<double> test_error = nnet::reduce_mean(
		nnet::sqrt<double>(nnet::varptr<double>(&expect_out) - test_res));
	auto xit = test_samples.begin();
	auto yit = test_out.begin();
	double total_err = 0;
	for (size_t i = 0; i < n_test_sample; ++i)
	{
		std::vector<double> xbatch(xit + i * n_in, xit + (i + 1) * n_in);
		std::vector<double> ybatch(yit + i * n_in / 2, yit + (i + 1) * n_in / 2);
		test_in = xbatch;
		expect_out = ybatch;

		double test_err = nnet::expose<double>(test_error)[0];
		total_err += test_err;
		std::cout << "test error at " << i << ": " << test_err << '\n';
	}
	std::cout << "total error " << total_err << '\n';
}

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	TestParams params;

	bool test_mnist;
	size_t n_simple_samples;

	options.desc_.add_options()
		("mnist", opt::bool_switch(&test_mnist)->default_value(false),
			"whether to take mnist data or not")
		("train", opt::bool_switch(&params.train)->default_value(true),
			"whether to train or not")
		("pretrain", opt::bool_switch(&params.pretrain)->default_value(true),
			"whether to pretrain or not")
		("pretrain_path", opt::value<std::string>(&params.pretrain_path)->default_value(""),
			"filename to save pretrained model")
		("save", opt::value<std::string>(&params.savepath)->default_value(""),
			"filename to save model")
		("load", opt::value<std::string>(&params.loadpath)->default_value(
			"rocnnet/pretrained/dbmodel.pbx"),
			"filename to load pretrained model")
		("k", opt::value(&params.n_cont_div)->default_value(15), "k-CD or k-PCD")
		("train_epoch", opt::value(&params.train_epochs)->default_value(1000),
			"epoch training iteration")
		("pretrain_epoch", opt::value(&params.pretrain_epochs)->default_value(1000),
			"epoch training iteration")
		("n_samples", opt::value(&n_simple_samples)->default_value(300),
			"epoch training iteration");

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

	if (test_mnist)
	{
		try
		{
			Py_Initialize();
			np::initialize();
			std::vector<ImgPtrT> datasets = get_mnist_data();

			ImgPtrT training_set = datasets[0];
			ImgPtrT valid_set = datasets[1];
			ImgPtrT testing_set = datasets[2];

			mnist_test(training_set, valid_set, testing_set, params);
		}
		catch(const bp::error_already_set&)
		{
			std::cerr << "[ERROR]: Uncaught python exception:\n";
			PyErr_Print();
			return 1;
		}
	}
	else
	{
		params.pretrain_epochs = 1;
		params.training_epochs = 1;
		simpler_test(n_simple_samples, n_simple_samples / 6, 10, params);
	}

	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
