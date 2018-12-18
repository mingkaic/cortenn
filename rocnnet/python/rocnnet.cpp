#include <fstream>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "rocnnet/eqns/activations.hpp"

#include "rocnnet/modl/fc_layer.hpp"
#include "rocnnet/modl/mlp.hpp"
#include "rocnnet/modl/conv_layer.hpp"

#include "rocnnet/trainer/gd_trainer.hpp"

namespace py = pybind11;

namespace pyrocnnet
{

modl::LayerInfo layerinfo_init (modl::HiddenFunc hidden, size_t n_out)
{
	return modl::LayerInfo{n_out, hidden};
}

modl::FCptrT fcl_init (std::vector<uint8_t> n_inputs, uint8_t n_output,
	std::string label)
{
	return modl::FCptrT(new modl::FCLayer(n_inputs, n_output, label));
}

modl::MLptrT mlp_init (uint8_t n_input, std::vector<modl::LayerInfo> layers,
	std::string label)
{
	return modl::MLptrT(new modl::MLP(n_input, layers, label));
}

eqns::ApproxFuncT get_sgd (double learning_rate)
{
	return [&](ade::TensptrT& root, eqns::VariablesT leaves)
	{
		return eqns::sgd(root, leaves, learning_rate);
	};
}

eqns::ApproxFuncT get_rms_momentum (double learning_rate,
	double discount_factor, double epsilon)
{
	return [&](ade::TensptrT& root, eqns::VariablesT leaves)
	{
		return eqns::rms_momentum(root, leaves, learning_rate,
			discount_factor, epsilon);
	};
}

}

PYBIND11_MODULE(rocnnet, m)
{
	m.doc() = "rocnnet api";

	py::class_<modl::iMarshaler,modl::MarsptrT> marshaler(m, "iMarshaler");
	py::class_<modl::FCLayer,modl::FCptrT> fcl(m, "FCLayer");
	py::class_<modl::MLP,modl::MLptrT> mlp(m, "MLP");

	py::implicitly_convertible<modl::iMarshaler,modl::FCLayer>();
	py::implicitly_convertible<modl::iMarshaler,modl::MLP>();

	py::class_<modl::LayerInfo> layerinfo(m, "LayerInfo");
	py::class_<GDTrainer> gdtrainer(m, "GDTrainer");

	// marshaler
	marshaler
		.def("serialize_to_string", [](py::object self, ade::TensptrT source)
		{
			std::stringstream savestr;
			modl::save(savestr, source, self.cast<modl::iMarshaler*>());
			return savestr.str();
		}, "load a version of this instance from a data");

	// fcl
	m.def("get_fcl", &pyrocnnet::fcl_init);
	fcl
		.def("copy", [](py::object self)
		{
			return std::make_shared<modl::FCLayer>(*self.cast<modl::FCLayer*>());
		}, "deep copy this instance")
		.def("parse_from_string", [](py::object self, std::string data)
		{
			auto out = std::make_shared<modl::FCLayer>(*self.cast<modl::FCLayer*>());
			std::stringstream loadstr;
			loadstr << data;
			modl::load(loadstr, out.get());
			return out;
		}, "load a version of this instance from a data")
		.def("forward", [](py::object self, ade::TensT inputs)
		{
			return (*self.cast<modl::FCLayer*>())(inputs);
		}, "forward input tensor and returned connected output");

	// mlp
	m.def("get_mlp", &pyrocnnet::mlp_init);
	mlp
		.def("copy", [](py::object self)
		{
			return std::make_shared<modl::MLP>(*self.cast<modl::MLP*>());
		}, "deep copy this instance")
		.def("parse_from_string", [](py::object self, std::string data)
		{
			auto out = std::make_shared<modl::MLP>(*self.cast<modl::MLP*>());
			std::stringstream loadstr;
			loadstr << data;
			modl::load(loadstr, out.get());
			return out;
		}, "load a version of this instance from a data")
		.def("forward", [](py::object self, ade::TensptrT input)
		{
			return (*self.cast<modl::MLP*>())(input);
		}, "forward input tensor and returned connected output");

	// layerinfo
	m.def("get_layer", &pyrocnnet::layerinfo_init);

	// gd_trainer
	gdtrainer
		.def(py::init<modl::MLptrT,eqns::ApproxFuncT,uint8_t,std::string>())
		.def("train", &GDTrainer::train, "train internal variables")
		.def("train_in", [](py::object self)
		{
			return self.cast<GDTrainer*>()->train_in_;
		}, "get train_in variable")
		.def("expected_out", [](py::object self)
		{
			return self.cast<GDTrainer*>()->expected_out_;
		}, "get expected_out variable")
		.def("error", [](py::object self)
		{
			return self.cast<GDTrainer*>()->error_;
		}, "get error output");


	// inlines
	m.def("sigmoid", &eqns::sigmoid);
	m.def("tanh", &eqns::tanh);
	m.def("softmax", &eqns::softmax);

	m.def("get_sgd", &pyrocnnet::get_sgd);
	m.def("get_rms_momentum", &pyrocnnet::get_rms_momentum);

	m.def("print_ptr", [](modl::MLptrT ptr){ std::cout << ptr.get() << std::endl; });
}
