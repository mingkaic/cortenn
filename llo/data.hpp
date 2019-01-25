///
/// data.hpp
/// llo
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include <memory>

#include "unsupported/Eigen/CXX11/Tensor"

#include "ade/coord.hpp"
#include "ade/ileaf.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

template <typename T>
using TensorT = Eigen::Tensor<T,ade::rank_cap,Eigen::RowMajor>;

template <typename T>
TensorT<T> get_tensor (T* data, const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr != data)
	{
		return Eigen::TensorMap<TensorT<T>>(data, slist);
	}
	TensorT<T> out(slist);
	out.setZero();
	return out;
}

template <typename T>
ade::Shape get_shape (const TensorT<T>& tens)
{
	auto slist = tens.dimensions();
	return ade::Shape(std::vector<ade::DimT>(slist.begin(), slist.end()));
}

template <typename T>
using TensptrT = std::shared_ptr<TensorT<T>>;

template <typename T>
TensptrT<T> get_tensorptr (T* data, const ade::Shape& shape)
{
	std::array<Eigen::Index,ade::rank_cap> slist;
	std::copy(shape.begin(), shape.end(), slist.begin());
	if (nullptr != data)
	{
		return std::make_shared<TensorT<T>>(
			Eigen::TensorMap<TensorT<T>>(data, slist));
	}
	auto out = std::make_shared<TensorT<T>>(slist);
	out->setZero();
	return out;
}

/// Data to pass around when evaluating
template <typename T>
struct DataArg
{
	TensptrT<T> data_;

	/// Coordinate mapper
	ade::CoordptrT mapper_;

	/// True if the coordinate mapper accepts input coordinates,
	/// False if it accepts output coordinates
	bool push_;
};

template <typename T>
using DataArgsT = std::vector<DataArg<T>>;

struct iVariable : public ade::iLeaf
{
	virtual ~iVariable (void) = default;

	virtual void assign (void* input,
		age::_GENERATED_DTYPE dtype, ade::Shape shape) = 0;

	virtual std::string get_label (void) const = 0;

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return this->get_label() + "(" + this->shape().to_string() + ")";
	}
};

#define TEMPCONVERT(TYPE)std::vector<T>((TYPE*) input, (TYPE*) input + n)

template <typename T>
TensptrT<T> raw_to_matrix (void* input,
	age::_GENERATED_DTYPE intype, const ade::Shape& shape)
{
	size_t n = shape.n_elems();
	std::vector<T> data;
	switch (intype)
	{
		case age::DOUBLE:
			data = TEMPCONVERT(double);
			break;
		case age::FLOAT:
			data = TEMPCONVERT(float);
			break;
		case age::INT8:
			data = TEMPCONVERT(int8_t);
			break;
		case age::INT16:
			data = TEMPCONVERT(int16_t);
			break;
		case age::INT32:
			data = TEMPCONVERT(int32_t);
			break;
		case age::INT64:
			data = TEMPCONVERT(int64_t);
			break;
		case age::UINT8:
			data = TEMPCONVERT(uint8_t);
			break;
		case age::UINT16:
			data = TEMPCONVERT(uint16_t);
			break;
		case age::UINT32:
			data = TEMPCONVERT(uint32_t);
			break;
		case age::UINT64:
			data = TEMPCONVERT(uint64_t);
			break;
		default:
			logs::fatalf("invalid input type %s",
				age::name_type(intype).c_str());
	}
	return get_tensorptr(data.data(), shape);
}

using iVarptrT = std::shared_ptr<iVariable>;

/// Leaf node containing data
template <typename T>
struct Variable final : public iVariable
{
	Variable (T* data, ade::Shape shape, std::string label) :
		label_(label), data_(get_tensor(data, shape)), shape_(shape) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

	Variable<T>& operator = (const Variable<T>& other) = default;

	Variable<T>& operator = (Variable<T>&& other) = default;

	/// Assign vectorized data to data source
	Variable<T>& operator = (std::vector<T> input)
	{
		size_t ninput = input.size();
		if (shape_.n_elems() != ninput)
		{
			logs::fatalf("cannot assign vector of %d elements to "
				"internal data of shape %s", ninput,
				shape().to_string().c_str());
		}
		std::memcpy(data_.data(), input.data(), ninput * sizeof(T));
		return *this;
	}

	Variable<T>& operator = (const TensorT<T>& input)
	{
		data_ = input;
		return *this;
	}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iLeaf
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iLeaf
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iLeaf
	size_t type_code (void) const override
	{
		return age::get_type<T>();
	}

	/// Implementation of iVariable
	void assign (void* input,
		age::_GENERATED_DTYPE dtype, ade::Shape shape) override
	{
		auto temp = raw_to_matrix<T>(input, dtype, shape);
		data_ = *temp;
	}

	/// Implementation of iVariable
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Return number of bytes in data source
	size_t nbytes (void) const
	{
		return sizeof(T) * shape_.n_elems();
	}

private:
	/// Label for distinguishing variable nodes
	std::string label_;

	TensorT<T> data_;

	ade::Shape shape_;
};

/// Smart pointer for variable nodes
template <typename T>
using VarptrT = std::shared_ptr<Variable<T>>;

/// Return new variable containing input vector data according to
/// specified shape and labelled according to input label
/// Throw error if the input vector size differs from shape.n_elems()
template <typename T>
VarptrT<T> get_variable (std::vector<T> data, ade::Shape shape,
	std::string label = "")
{
	if (data.size() != shape.n_elems())
	{
		logs::fatalf("cannot create variable with data size %d "
			"against shape %s", data.size(), shape.to_string().c_str());
	}
	return std::make_shared<Variable<T>>(data.data(), shape, label);
}

/// Return new variable containing 0s according to
/// specified shape and labelled according to input label
template <typename T>
VarptrT<T> get_variable (ade::Shape shape, std::string label = "")
{
	return get_variable(std::vector<T>(shape.n_elems(), 0), shape, label);
}

/// Return new variable containing 0s according to
/// specified shape and labelled according to input label
template <typename T>
VarptrT<T> get_scalar (T scalar, ade::Shape shape, std::string label = "")
{
	if (label.empty())
	{
		label = fmts::to_string(scalar);
	}
	return llo::get_variable(std::vector<T>(shape.n_elems(),scalar),
		shape, label);
}

}

#endif // LLO_DATA_HPP
