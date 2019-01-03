#include <memory>

#include "ade/ileaf.hpp"

#include "llo/generated/codes.hpp"

#include "llo/operator.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

/// GenericData for holding data when passing up the tensor graph
struct GenericData final
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, age::_GENERATED_DTYPE dtype);

	/// Copy over data of specified type while retaining shape
	/// This makes the assumption that the indata fits in shape perfectly
	void copyover (const char* indata, age::_GENERATED_DTYPE intype);

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

	/// Assign vectorized data to source
	template <typename T>
	Variable& operator = (std::vector<T> data)
	{
		GenericRef ref((char*) &data[0], shape(), age::get_type<T>());
		return operator = (ref);
	}

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

	void* data (void) override
	{
		return data_.data_.get();
	}

	const void* data (void) const override
	{
		return data_.data_.get();
	}

	size_t type_code (void) const override
	{
		return data_.dtype_;
	}

	size_t nbytes (void) const
	{
		return type_size(data_.dtype_) * data_.shape_.n_elems();
	}

	std::string label_;

private:
	GenericData data_;
};

using VarptrT = std::shared_ptr<llo::Variable>;

template <typename T>
VarptrT get_variable (std::vector<T> data, ade::Shape shape,
	std::string label = "")
{
	if (data.size() != shape.n_elems())
	{
		logs::fatalf("cannot create variable with data size %d "
			"against shape %s", data.size(), shape.to_string().c_str());
	}
	return VarptrT(new Variable((char*) &data[0],
		age::get_type<T>(), shape, label));
}

template <typename T>
VarptrT get_variable (ade::Shape shape, std::string label = "")
{
	return get_variable(std::vector<T>(shape.n_elems(), 0), shape, label);
}

template <typename T>
VarptrT data (T scalar, ade::Shape shape, std::string label = "")
{
	return llo::get_variable(std::vector<T>(shape.n_elems(),scalar),
		shape, label);
}

struct DataArg
{
	std::shared_ptr<char> data_;

	ade::Shape shape_;

	ade::CoordptrT mapper_;

	bool fwd_;
};

using DataArgsT = std::vector<DataArg>;

template <typename T>
VecRef<T> to_ref (DataArg& arg)
{
	return VecRef<T>{
		(const T*) arg.data_.get(),
		arg.shape_,
		arg.mapper_,
		arg.fwd_,
	};
}

template <typename T>
std::vector<VecRef<T>> to_refs (DataArgsT& args)
{
	std::vector<VecRef<T>> out;
	std::transform(args.begin(), args.end(), std::back_inserter(out),
		[](DataArg& arg)
		{
			return to_ref<T>(arg);
		});
	return out;
}

}

#endif // LLO_DATA_HPP
