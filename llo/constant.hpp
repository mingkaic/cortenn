///
/// constant.hpp
/// llo
///
/// Purpose:
/// Define constant for storing generic data
///

#include "llo/generated/codes.hpp"
#include "llo/generated/opmap.hpp"

#include "ade/ileaf.hpp"

#ifndef LLO_CONSTANT_HPP
#define LLO_CONSTANT_HPP

namespace llo
{

struct Constant final : public ade::iLeaf
{
	static Constant* get (const char* data,
		age::_GENERATED_DTYPE dtype, ade::Shape shape)
	{
		return new Constant(data, dtype, shape);
	}

	template <typename T>
	static Constant* get (T scalar, ade::Shape shape)
	{
		size_t n = shape.n_elems();
		T buffer[n];
		std::fill(buffer, buffer + n, scalar);
		return new Constant((char*) buffer, age::get_type<T>(), shape);
	}

	Constant (const Constant& other) = delete;

	Constant (Constant&& other) = delete;

	Constant& operator = (const Constant& other) = delete;

	Constant& operator = (Constant&& other) = delete;

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return fmts::to_string(at<double>(0));
	}

	/// Implementation of iLeaf
	void* data (void) override
	{
		return &data_[0];
	}

	/// Implementation of iLeaf
	const void* data (void) const override
	{
		return data_.c_str();
	}

	/// Implementation of iLeaf
	size_t type_code (void) const override
	{
		return dtype_;
	}

	/// Implementation of iLeaf
	std::string type_label (void) const override
	{
		return age::name_type(dtype_);
	}

	virtual size_t nbytes (void) const override
	{
		return shape_.n_elems() * age::type_size(dtype_);
	}

	template <typename T>
	T at (size_t i) const
	{
		std::vector<T> out;
		age::type_convert(out, (void*) &data_[i * age::type_size(dtype_)],
			dtype_, 1);
		return out[0];
	}

private:
	Constant (const char* data, age::_GENERATED_DTYPE dtype, ade::Shape shape) :
		data_(data, shape.n_elems() * age::type_size(dtype)),
		shape_(shape), dtype_(dtype) {}

	/// Smartpointer to a block of untyped data
	std::string data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Type encoding of data_
	age::_GENERATED_DTYPE dtype_;
};

}

#endif // LLO_CONSTANT_HPP
