///
/// data.hpp
/// llo
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include <memory>

#include "ade/coord.hpp"
#include "ade/ileaf.hpp"

#include "llo/generated/codes.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

struct CDeleter final
{
	void operator () (void* p)
	{
		free(p);
	}
};

#define CONVERT(INTYPE)\
{\
	const INTYPE* ptr = (const INTYPE*) indata;\
	std::vector<T> temp(ptr, ptr + n);\
	std::memcpy(data_.get(), temp.data(), nbytes);\
} break;

template <typename T>
struct TypedData final
{
	TypedData (void) = default;

	TypedData (ade::Shape shape):
		data_((T*) malloc(shape.n_elems() * sizeof(T)),
			CDeleter()), shape_(shape) {}

	/// Copy over data of specified type while retaining shape
	/// This makes the assumption that the indata fits in shape perfectly
	void copyover (const char* indata, age::_GENERATED_DTYPE intype)
	{
		size_t n = shape_.n_elems();
		size_t nbytes = sizeof(T) * n;
		switch (intype)
		{
			case age::DOUBLE: CONVERT(double)
			case age::FLOAT: CONVERT(float)
			case age::INT8: CONVERT(int8_t)
			case age::INT16: CONVERT(int16_t)
			case age::INT32: CONVERT(int32_t)
			case age::INT64: CONVERT(int64_t)
			case age::UINT8: CONVERT(uint8_t)
			case age::UINT16: CONVERT(uint16_t)
			case age::UINT32: CONVERT(uint32_t)
			case age::UINT64: CONVERT(uint64_t)
			default: logs::fatalf("invalid input type %s",
				age::name_type(intype).c_str());
		}
	}

	/// Smartpointer to a block of typed data
	std::shared_ptr<T> data_;

	/// Shape of data_
	ade::Shape shape_;
};

/// Data to pass around when evaluating
template <typename T>
struct DataArg
{
	/// Smart pointer to generic data as bytes
	std::shared_ptr<T> data_;

	/// Shape of the generic data
	ade::Shape shape_;

	/// Coordinate mapper
	ade::CoordptrT mapper_;

	/// True if the coordinate mapper accepts input coordinates,
	/// False if it accepts output coordinates
	bool push_;
};

/// Vector of DataArgs to hold arguments
template <typename T>
using DataArgsT = std::vector<DataArg<T>>;

/// GenericData for holding data when passing up the tensor graph
struct GenericData final
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, age::_GENERATED_DTYPE dtype) :
		data_((char*) malloc(shape.n_elems() * type_size(dtype)),
			CDeleter()), shape_(shape), dtype_(dtype) {}

	/// Smartpointer to a block of untyped data
	std::shared_ptr<char> data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Type encoding of data_
	age::_GENERATED_DTYPE dtype_;
};

/// GenericRef for holding data
/// Ref uses raw pointer instead of shared, so it's memory unsafe
struct GenericRef
{
	GenericRef (char* data, ade::Shape shape, age::_GENERATED_DTYPE dtype) :
		data_(data), shape_(shape), dtype_(dtype) {}

	GenericRef (GenericData& generic) :
		data_(generic.data_.get()),
		shape_(generic.shape_), dtype_(generic.dtype_) {}

	/// Raw pointer to a block of untyped data
	char* data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Data type of data_
	age::_GENERATED_DTYPE dtype_;
};

/// Leaf node containing GenericData
struct Variable final : public ade::iLeaf
{
	Variable (const char* data, age::_GENERATED_DTYPE dtype,
		ade::Shape shape, std::string label) :
		label_(label), data_(shape, dtype)
	{
		if (nullptr != data)
		{
			std::memcpy(data_.data_.get(), data, nbytes());
		}
		else
		{
			std::memset(data_.data_.get(), 0, nbytes());
		}
	}

	Variable (const Variable& other) :
		label_(other.label_), data_(other.shape(), (age::_GENERATED_DTYPE) other.type_code())
	{
		std::memcpy((char*) data_.data_.get(), (const char*) other.data(), nbytes());
	}

	Variable (Variable&& other) :
		label_(std::move(other.label_)), data_(std::move(other.data_)) {}

	Variable& operator = (const Variable& other)
	{
		if (this != &other)
		{
			label_ = other.label_;
			data_ = GenericData(other.shape(), (age::_GENERATED_DTYPE) other.type_code());
			std::memcpy((char*) data_.data_.get(), (const char*) other.data(), nbytes());
		}
		return *this;
	}

	Variable& operator = (Variable&& other)
	{
		if (this != &other)
		{
			label_ = std::move(other.label_);
			data_ = std::move(other.data_);
		}
		return *this;
	}

	/// Assign vectorized data to data source
	template <typename T>
	Variable& operator = (std::vector<T> data)
	{
		GenericRef ref((char*) data.data(), shape(), age::get_type<T>());
		return operator = (ref);
	}

	/// Assign generic reference to data source
	Variable& operator = (GenericRef data)
	{
		if (false == data.shape_.compatible_after(shape(), 0))
		{
			logs::fatalf("cannot assign data of incompatible shaped %s to "
				"internal data of shape %s", data.shape_.to_string().c_str(),
				shape().to_string().c_str());
		}
		if (data.dtype_ != data_.dtype_)
		{
			logs::fatalf("cannot assign data of incompatible types %s "
				"(external) and %s (internal)",
				age::name_type(data.dtype_).c_str(), age::name_type(data_.dtype_).c_str());
		}
		std::memcpy(data_.data_.get(), data.data_, nbytes());
		return *this;
	}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return data_.shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return label_ + "(" + data_.shape_.to_string() + ")";
	}

	/// Implementation of iLeaf
	void* data (void) override
	{
		return data_.data_.get();
	}

	/// Implementation of iLeaf
	const void* data (void) const override
	{
		return data_.data_.get();
	}

	/// Implementation of iLeaf
	size_t type_code (void) const override
	{
		return data_.dtype_;
	}

	/// Return number of bytes in data source
	size_t nbytes (void) const
	{
		return type_size(data_.dtype_) * data_.shape_.n_elems();
	}

	/// Label for distinguishing variable nodes
	std::string label_;

private:
	/// Generic data source
	GenericData data_;
};

/// Smart pointer for variable nodes
using VarptrT = std::shared_ptr<Variable>;

/// Return new variable containing input vector data according to
/// specified shape and labelled according to input label
/// Throw error if the input vector size differs from shape.n_elems()
template <typename T>
VarptrT get_variable (std::vector<T> data, ade::Shape shape,
	std::string label = "")
{
	if (data.size() != shape.n_elems())
	{
		logs::fatalf("cannot create variable with data size %d "
			"against shape %s", data.size(), shape.to_string().c_str());
	}
	return VarptrT(new Variable((char*) data.data(),
		age::get_type<T>(), shape, label));
}

/// Return new variable containing 0s according to
/// specified shape and labelled according to input label
template <typename T>
VarptrT get_variable (ade::Shape shape, std::string label = "")
{
	return get_variable(std::vector<T>(shape.n_elems(), 0), shape, label);
}

/// Return new variable containing 0s according to
/// specified shape and labelled according to input label
template <typename T>
VarptrT get_scalar (T scalar, ade::Shape shape, std::string label = "")
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
