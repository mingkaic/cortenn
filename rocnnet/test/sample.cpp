#include <iostream>
#include <fstream>

#include "dbg/ade.hpp"

#include "llo/shear.hpp"

#include "pbm/save.hpp"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/modl/gd_trainer.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	std::string outdir = "/tmp";
	std::string serialname = "gd_test.pbx";
	std::string serialpath = outdir + "/" + serialname;

	uint8_t n_in = 10;
	uint8_t n_out = n_in / 2;
	std::vector<LayerInfo> hiddens = {
		// use same sigmoid in static memory once models copy is established
		LayerInfo{9, sigmoid},
		LayerInfo{n_out, sigmoid}
	};
	MLP brain(n_in, hiddens, "brain");

	uint8_t n_batch = 3;
	ApproxFuncT approx = [](ade::TensptrT& root, VariablesT leaves)
	{
		return sgd(root, leaves, 0.9); // learning rate = 0.9
	};
	GDTrainer trainer(brain, approx, n_batch, "gdn");

	auto vars = brain.get_variables();
	// trainer.train_in_ = std::vector<double>{
	// 	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	// 	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	// 	1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	ade::TensptrT root = trainer.error_;

	for (auto var : vars)
	{
		auto der = llo::derive(root, var);
		auto pruned = llo::zero_prune(der);

		llo::GenericData gdata = llo::eval(pruned, age::DOUBLE);
		double* gdptr = (double*) gdata.data_.get();
		std::cout << err::to_string(gdptr, gdptr + gdata.shape_.n_elems()) << '\n';
	}

	llo::GenericData data = llo::eval(root, age::DOUBLE);
	double* dptr = (double*) data.data_.get();
	std::cout << err::to_string(dptr, dptr + data.shape_.n_elems()) << '\n';

	tenncor::Graph graph;
	std::vector<ade::TensptrT> roots = {trainer.error_};
	save_graph(graph, roots);
	std::fstream outstr(serialpath, std::ios::out | std::ios::trunc | std::ios::binary);
	if (!graph.SerializeToOstream(&outstr))
	{
		err::warn("failed to serialize initial trainer");
	}

	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
