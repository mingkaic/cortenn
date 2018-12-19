#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "llo/data.hpp"
#include "llo/eval.hpp"
#include "llo/zprune.hpp"

namespace py = pybind11;

namespace pyllo
{

llo::VarptrT variable (py::array data, std::string label)
{
	py::buffer_info info = data.request();
	ade::Shape shape(std::vector<ade::DimT>(
		info.shape.begin(), info.shape.end()));
	size_t n = shape.n_elems();
	char kind = data.dtype().kind();
	switch (kind)
	{
		case 'f':
		{
			double* dptr = static_cast<double*>(info.ptr);
			return llo::get_variable(std::vector<double>(dptr, dptr + n),
				shape, label);
		}
			break;
		case 'i':
		{
			int64_t* dptr = static_cast<int64_t*>(info.ptr);
			return llo::get_variable(std::vector<int64_t>(dptr, dptr + n),
				shape, label);
		}
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
}

void assign (llo::Variable* target, py::array data)
{
	py::buffer_info info = data.request();
	ade::Shape shape(std::vector<ade::DimT>(
		info.shape.begin(), info.shape.end()));
	size_t n = shape.n_elems();
	char kind = data.dtype().kind();
	switch (kind)
	{
		case 'f':
		{
			double* dptr = static_cast<double*>(info.ptr);
			*target = std::vector<double>(dptr, dptr + n);
		}
			break;
		case 'i':
		{
			int64_t* dptr = static_cast<int64_t*>(info.ptr);
			*target = std::vector<int64_t>(dptr, dptr + n);
		}
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
}

py::array evaluate (ade::TensptrT tens,
	py::dtype dtype = py::dtype::of<double>())
{
	age::_GENERATED_DTYPE ctype = age::BAD_TYPE;
	char kind = dtype.kind();
	switch (kind)
	{
		case 'f':
			ctype = age::DOUBLE;
			break;
		case 'i':
			ctype = age::INT64;
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
	llo::GenericData gdata = llo::eval(tens, ctype);
	void* vptr = gdata.data_.get();
	auto it = gdata.shape_.begin();
	auto et = gdata.shape_.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	return py::array(dtype, py::array::ShapeContainer(it, et), vptr);
}

void seed_engine (size_t seed)
{
	llo::get_engine().seed(seed);
}

}

PYBIND11_MODULE(llo, m)
{
	m.doc() = "llo variables";

	py::object tensor = (py::object) 
		py::module::import("llo.age").attr("Tensor");
	py::class_<llo::Variable,llo::VarptrT> variable(m, "Variable", tensor);

	py::implicitly_convertible<ade::iTensor,llo::Variable>();

	// variable
	m.def("variable", &pyllo::variable, "create tensor variable");
	variable
		.def("assign", [](py::object self, py::array data)
		{
			pyllo::assign(self.cast<llo::Variable*>(), data);
		}, "assign to variable");


	// inline
	m.def("evaluate", &pyllo::evaluate, "evaluate tensor",
		py::arg("tens"), py::arg("dtype") = py::dtype::of<double>(),
		"evaluate data of tens according to dtype");
	m.def("derive", llo::derive,
		"derive tensor with respect to some derive");
	m.def("seed", &pyllo::seed_engine, "seed internal rng");
}
