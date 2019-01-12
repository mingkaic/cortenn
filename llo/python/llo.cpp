#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "ade/ade.hpp"

#include "llo/data.hpp"
#include "llo/eval.hpp"
#include "llo/opt/derive.hpp"

#include "dbg/ade.hpp"

namespace py = pybind11;

namespace pyllo
{

ade::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return ade::Shape(std::vector<ade::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

std::vector<ade::DimT> c2pshape (ade::Shape& cshape)
{
	auto it = cshape.begin();
	auto et = cshape.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	std::vector<ade::DimT> fwd(it, et);
	return std::vector<ade::DimT>(fwd.rbegin(), fwd.rend());
}

llo::VarptrT variable (py::array data, std::string label)
{
	py::buffer_info info = data.request();
	ade::Shape shape = p2cshape(info.shape);
	size_t n = shape.n_elems();
	auto dtype = data.dtype();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	switch (kind)
	{
		case 'f':
			switch (tbytes)
			{
				case 4: // float32
				{
					float* dptr = static_cast<float*>(info.ptr);
					return llo::get_variable(std::vector<float>(dptr, dptr + n),
						shape, label);
				}
				case 8: // float64
				{
					double* dptr = static_cast<double*>(info.ptr);
					return llo::get_variable(std::vector<double>(dptr, dptr + n),
						shape, label);
				}
				default:
					logs::fatalf("unsupported float type with %d bytes", tbytes);
			}
			break;
		case 'i':
			switch (tbytes)
			{
				case 1: // int8
				{
					int8_t* dptr = static_cast<int8_t*>(info.ptr);
					return llo::get_variable(std::vector<int8_t>(dptr, dptr + n),
						shape, label);
				}
				case 2: // int16
				{
					int16_t* dptr = static_cast<int16_t*>(info.ptr);
					return llo::get_variable(std::vector<int16_t>(dptr, dptr + n),
						shape, label);
				}
				case 4: // int32
				{
					int32_t* dptr = static_cast<int32_t*>(info.ptr);
					return llo::get_variable(std::vector<int32_t>(dptr, dptr + n),
						shape, label);
				}
				case 8: // int64
				{
					int64_t* dptr = static_cast<int64_t*>(info.ptr);
					return llo::get_variable(std::vector<int64_t>(dptr, dptr + n),
						shape, label);
				}
				default:
					logs::fatalf("unsupported integer type with %d bytes", tbytes);
			}
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
}

void assign (llo::Variable* target, py::array data)
{
	py::buffer_info info = data.request();
	ade::Shape shape = p2cshape(info.shape);
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
	auto pshape = c2pshape(gdata.shape_);
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()), vptr);
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

	((py::class_<ade::iTensor,ade::TensptrT>) tensor)
		.def("__str__", &ade::iTensor::to_string)
		.def("shape", [](py::object self)
		{
			ade::Shape shape = self.cast<ade::iTensor*>()->shape();
			auto pshape = pyllo::c2pshape(shape);
			std::vector<int> ipshape(pshape.begin(), pshape.end());
			return py::array(ipshape.size(), &ipshape[0]);
		})
		.def("children", [](py::object self)
		{
			std::vector<ade::TensptrT> tens;
			if (auto f = dynamic_cast<ade::iFunctor*>(
				self.cast<ade::iTensor*>()))
			{
				auto args = f->get_children();
				std::transform(args.begin(), args.end(),
				std::back_inserter(tens),
				[](ade::MappedTensor& mten)
				{
					return mten.get_tensor();
				});
			}
			return tens;
		});

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
	m.def("print_graph", [](ade::TensptrT root, bool showshape)
		{
			PrettyEquation peq;
			peq.showshape_ = showshape;
			peq.print(std::cout, root);
		}, "print graph of root tensor",
		py::arg("root"), py::arg("showshape") = false);
}
