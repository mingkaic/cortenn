///
/// variable.hpp
/// llo
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include "ade/coord.hpp"
#include "ade/ileaf.hpp"

#include "llo/tensor.hpp"

#ifndef LLO_VARIABLE_HPP
#define LLO_VARIABLE_HPP

namespace llo
{

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

/// Leaf node containing data
template <typename T>
struct Variable final : public iVariable
{
	static Variable* get (ade::Shape shape, std::string label = "")
	{
		return Variable<T>::get(std::vector<T>(shape.n_elems(), 0),
			shape, label);
	}

	static Variable* get (T* ptr, ade::Shape shape, std::string label = "")
	{
		return new Variable<T>(ptr, shape, label);
	}

	static Variable* get (T scalar, ade::Shape shape, std::string label = "")
	{
		if (label.empty())
		{
			label = fmts::to_string(scalar);
		}
		return Variable<T>::get(std::vector<T>(shape.n_elems(),scalar),
			shape, label);
	}

	static Variable* get (std::vector<T> data, ade::Shape shape,
		std::string label = "")
	{
		if (data.size() != shape.n_elems())
		{
			logs::fatalf("cannot create variable with data size %d "
				"against shape %s", data.size(), shape.to_string().c_str());
		}
		return new Variable<T>(data.data(), shape, label);
	}

	static Variable* get (const Variable& other)
	{
		return new Variable<T>(other);
	}

	static Variable* get (Variable&& other)
	{
		return new Variable<T>(std::move(other));
	}

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

	/// Implementation of iLeaf
	std::string type_label (void) const override
	{
		return age::name_type(age::get_type<T>());
	}

	/// Implementation of iVariable
	void assign (void* input,
		age::_GENERATED_DTYPE dtype, ade::Shape shape) override
	{
		auto temp = raw_to_tensorptr<T>(input, dtype, shape);
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
	Variable (T* data, ade::Shape shape, std::string label) :
		label_(label), data_(get_tensor(data, shape)), shape_(shape) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

	/// Label for distinguishing variable nodes
	std::string label_;

	TensorT<T> data_;

	ade::Shape shape_;
};

/// Smart pointer for variable nodes
template <typename T>
using VarptrT = std::shared_ptr<Variable<T>>;

}

#endif // LLO_VARIABLE_HPP
