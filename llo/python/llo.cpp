#include <fstream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "ade/ade.hpp"

#include "llo/variable.hpp"
#include "llo/eval.hpp"
#include "llo/opt/derive.hpp"

namespace py = pybind11;

namespace pyllo
{

ade::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return ade::Shape(std::vector<ade::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

std::vector<ade::DimT> c2pshape (const ade::Shape& cshape)
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

std::vector<ade::DimT> c2pshape (
	const Eigen::DSizes<long,ade::rank_cap>& cshape)
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

template <typename T>
py::array typedata_to_array (llo::TensptrT<T>& tdata, py::dtype dtype)
{
	auto pshape = pyllo::c2pshape(tdata->dimensions());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tdata->data());
}

}

PYBIND11_MODULE(llo, m)
{
	m.doc() = "llo variables";

	// ==== tensor ====
	py::object tensor = (py::object)
		py::module::import("llo.age").attr("Tensor");

	((py::class_<ade::iTensor,ade::TensptrT>) tensor)
		.def("__str__",
		&ade::iTensor::to_string,
		"Return string representation of this tensor instance")
		.def("shape",
		[](py::object self)
		{
			ade::Shape shape = self.cast<ade::iTensor*>()->shape();
			auto pshape = pyllo::c2pshape(shape);
			std::vector<int> ipshape(pshape.begin(), pshape.end());
			return py::array(ipshape.size(), ipshape.data());
		},
		"Return this instance's shape")
		.def("children",
		[](py::object self)
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
		},
		"Return this instance's tensor arguments")
		.def("evaluate",
		[](py::object self, py::dtype dtype)
		{
			char kind = dtype.kind();
			switch (kind)
			{
				case 'f':
				{
					auto gdata = llo::eval<double>(self.cast<ade::iTensor*>());
					return pyllo::typedata_to_array(gdata, dtype);
				}
				case 'i':
				{
					auto gdata = llo::eval<int64_t>(self.cast<ade::iTensor*>());
					return pyllo::typedata_to_array(gdata, dtype);
				}
				default:
					logs::fatalf("unknown dtype %c", kind);
			}
		},
		"Return calculated data",
		py::arg("dtype") = py::dtype::of<double>());

	// ==== variable ====
	py::class_<llo::iVariable,std::shared_ptr<llo::iVariable>> variable(
		m, "Variable", tensor);

	py::implicitly_convertible<ade::iTensor,llo::iVariable>();

	m.def("variable",
	[](py::array data, std::string label) -> std::shared_ptr<llo::iVariable>
	{
		py::buffer_info info = data.request();
		ade::Shape shape = pyllo::p2cshape(info.shape);
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
						return std::shared_ptr<llo::iVariable>(
							llo::Variable<float>::get(
								std::vector<float>(dptr, dptr + n),
								shape, label));
					}
					case 8: // float64
					{
						double* dptr = static_cast<double*>(info.ptr);
						return std::shared_ptr<llo::iVariable>(
							llo::Variable<double>::get(
								std::vector<double>(dptr, dptr + n),
								shape, label));
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
						return std::shared_ptr<llo::iVariable>(
							llo::Variable<int8_t>::get(
								std::vector<int8_t>(dptr, dptr + n),
								shape, label));
					}
					case 2: // int16
					{
						int16_t* dptr = static_cast<int16_t*>(info.ptr);
						return std::shared_ptr<llo::iVariable>(
							llo::Variable<int16_t>::get(
								std::vector<int16_t>(dptr, dptr + n),
								shape, label));
					}
					case 4: // int32
					{
						int32_t* dptr = static_cast<int32_t*>(info.ptr);
						return std::shared_ptr<llo::iVariable>(
							llo::Variable<int32_t>::get(
								std::vector<int32_t>(dptr, dptr + n),
								shape, label));
					}
					case 8: // int64
					{
						int64_t* dptr = static_cast<int64_t*>(info.ptr);
						return std::shared_ptr<llo::iVariable>(
							llo::Variable<int64_t>::get(
								std::vector<int64_t>(dptr, dptr + n),
								shape, label));
					}
					default:
						logs::fatalf("unsupported integer type with %d bytes", tbytes);
				}
				break;
			default:
				logs::fatalf("unknown dtype %c", kind);
		}
	},
	"Return labelled variable containing numpy data array");

	variable
		.def("assign",
		[](py::object self, py::array data)
		{
			llo::iVariable* target = self.cast<llo::iVariable*>();
			py::buffer_info info = data.request();
			ade::Shape shape = pyllo::p2cshape(info.shape);
			auto dtype = data.dtype();
			char kind = dtype.kind();
			age::_GENERATED_DTYPE age_dtype = age::BAD_TYPE;
			py::ssize_t tbytes = dtype.itemsize();
			switch (kind)
			{
				case 'f':
					switch (tbytes)
					{
						case 4: // float32
							age_dtype = age::FLOAT;
							break;
						case 8: // float64
							age_dtype = age::DOUBLE;
							break;
						default:
							logs::fatalf("unsupported float type with %d bytes", tbytes);
					}
					break;
				case 'i':
					switch (tbytes)
					{
						case 1: // int8
							age_dtype = age::INT8;
							break;
						case 2: // int16
							age_dtype = age::INT16;
							break;
						case 4: // int32
							age_dtype = age::INT32;
							break;
						case 8: // int64
							age_dtype = age::INT64;
							break;
						default:
							logs::fatalf("unsupported integer type with %d bytes", tbytes);
					}
					break;
				default:
					logs::fatalf("unknown dtype %c", kind);
			}
			target->assign(info.ptr, age_dtype, shape);
		},
		"Assign numpy data array to variable");

	// ==== inline functions ====
	m.def("derive",
	llo::derive,
	"Return derivative of first tensor with respect to second tensor");

	m.def("seed",
	[](size_t seed)
	{
		llo::get_engine().seed(seed);
	},
	"Seed internal RNG");
}
