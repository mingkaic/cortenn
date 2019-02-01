
#include "unsupported/Eigen/CXX11/Tensor"

#include "ade/shape.hpp"

#include "llo/generated/data.hpp"

#ifndef LLO_TENSOR_HPP
#define LLO_TENSOR_HPP

namespace llo
{

template <typename T>
using TensorT = Eigen::Tensor<T,ade::rank_cap,Eigen::RowMajor>;

template <typename T>
using TensptrT = std::shared_ptr<TensorT<T>>;

template <typename T>
inline TensorT<T> get_tensor (T* data, const ade::Shape& shape)
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

template <typename T>
TensptrT<T> raw_to_tensorptr (void* input,
	age::_GENERATED_DTYPE intype, const ade::Shape& shape)
{
	std::vector<T> data;
	age::type_convert(data, input, intype, shape.n_elems());
	return get_tensorptr(data.data(), shape);
}

template <typename T>
ade::Shape get_shape (const TensorT<T>& tens)
{
	auto slist = tens.dimensions();
	return ade::Shape(std::vector<ade::DimT>(slist.begin(), slist.end()));
}

}

#endif // LLO_TENSOR_HPP
