///
/// operator.hpp
/// llo
///
/// Purpose:
/// Define functions manipulating tensor data values
/// No function in this file makes any attempt to check for nullptrs
///

#include <cstring>
#include <cmath>
#include <functional>
#include <random>

#include "Eigen/Core"

#include "llo/data.hpp"

#ifndef LLO_OPERATOR_HPP
#define LLO_OPERATOR_HPP

namespace llo
{

/// RNG engine used
using EngineT = std::default_random_engine;

/// Return global random generator
EngineT& get_engine (void);

bool is_identity (ade::CoordptrT& coorder);

/// Generic unary operation assuming identity mapping
template <typename T>
void unary (T* out, ade::Shape& outshape,
	DataArg<T> in, std::function<T(const T&)> f)
{
	T* inptr = in.data_.get();
	if (is_identity(in.mapper_))
	{
		for (ade::NElemT i = 0, n = in.shape_.n_elems(); i < n; ++i)
		{
			out[i] = f(inptr[i]);
		}
	}
	else
	{
		ade::CoordT coord;
		if (in.push_)
		{
			for (ade::NElemT i = 0, n = in.shape_.n_elems(); i < n; ++i)
			{
				in.mapper_->forward(coord.begin(),
					ade::coordinate(in.shape_, i).begin());
				out[ade::index(outshape, coord)] = f(inptr[i]);
			}
		}
		else
		{
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				in.mapper_->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				out[i] = f(inptr[ade::index(in.shape_, coord)]);
			}
		}
	}
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
void abs (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::abs(src); });
}

template <>
void abs<uint8_t> (uint8_t* out, DataArg<uint8_t> in);

template <>
void abs<uint16_t> (uint16_t* out, DataArg<uint16_t> in);

template <>
void abs<uint32_t> (uint32_t* out, DataArg<uint32_t> in);

template <>
void abs<uint64_t> (uint64_t* out, DataArg<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
void neg (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return -src; });
}

template <>
void neg<uint8_t> (uint8_t* out, DataArg<uint8_t> in);

template <>
void neg<uint16_t> (uint16_t* out, DataArg<uint16_t> in);

template <>
void neg<uint32_t> (uint32_t* out, DataArg<uint32_t> in);

template <>
void neg<uint64_t> (uint64_t* out, DataArg<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take bitwise nots of inputs
template <typename T>
void bit_not (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return !src; });
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
void sin (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::sin(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
void cos (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::cos(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
void tan (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::tan(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
void exp (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::exp(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
void log (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::log(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
void sqrt (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::sqrt(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
void round (T* out, DataArg<T> in)
{
	unary<T>(out, in.shape_, in, [](const T& src) { return std::round(src); });
}

/// Generic binary operation assuming identity mapping
template <typename T>
void binary (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b,
	std::function<T(const T&,const T&)> f)
{
	T* aptr = a.data_.get();
	T* bptr = b.data_.get();

	if (is_identity(a.mapper_))
	{
		std::memcpy(out, aptr, sizeof(T) * a.shape_.n_elems());
	}
	else
	{
		ade::CoordT coord;
		if (a.push_)
		{
			for (ade::NElemT i = 0, n = a.shape_.n_elems(); i < n; ++i)
			{
				a.mapper_->forward(coord.begin(),
					ade::coordinate(a.shape_, i).begin());
				out[ade::index(outshape, coord)] = aptr[i];
			}
		}
		else
		{
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				a.mapper_->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				out[i] = aptr[ade::index(a.shape_, coord)];
			}
		}
	}

	if (is_identity(b.mapper_))
	{
		for (ade::NElemT i = 0, n = b.shape_.n_elems(); i < n; ++i)
		{
			out[i] = f(out[i], bptr[i]);
		}
	}
	else
	{
		ade::CoordT coord;
		if (b.push_)
		{
			for (ade::NElemT i = 0, n = b.shape_.n_elems(); i < n; ++i)
			{
				b.mapper_->forward(coord.begin(),
					ade::coordinate(b.shape_, i).begin());
				out[ade::index(outshape, coord)] = f(
					out[ade::index(outshape, coord)], bptr[i]);
			}
		}
		else
		{
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				b.mapper_->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				out[i] = f(out[i], bptr[ade::index(b.shape_, coord)]);
			}
		}
	}
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
void pow (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& b, const T& x) { return std::pow(b, x); });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
void sub (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a - b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
void div (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a / b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
void eq (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a == b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply != operator
/// Only accept 2 arguments
template <typename T>
void neq (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a != b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply < operator
/// Only accept 2 arguments
template <typename T>
void lt (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a < b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply > operator
/// Only accept 2 arguments
template <typename T>
void gt (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a > b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T>
void rand_uniform (T* out,
	ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, outshape, a, b,
	[](const T& a, const T& b)
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_uniform<double> (double* out,
	ade::Shape& outshape, DataArg<double> a, DataArg<double> b);

template <>
void rand_uniform<float> (float* out,
	ade::Shape& outshape, DataArg<float> a, DataArg<float> b);

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::normal_distribution function
/// Only accept 2 arguments
template <typename T>
void rand_normal (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<double> (double* out,
	ade::Shape& outshape, DataArg<double> a, DataArg<double> b);

template <>
void rand_normal<float> (float* out,
	ade::Shape& outshape, DataArg<float> a, DataArg<float> b);

/// Generic n-nary operation
template <typename T>
void nnary (T* out, ade::Shape& outshape, DataArgsT<T> args,
	std::function<void(T&,const T&)> acc)
{
	ade::NElemT nout = outshape.n_elems();
	bool visited[nout];
	std::memset(visited, false, nout);
	ade::CoordT coord;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		DataArg<T>& arg = args[i];
		T* argptr = arg.data_.get();
		if (is_identity(arg.mapper_))
		{
			if (i == 0)
			{
				std::memcpy(out, argptr, sizeof(T) * arg.shape_.n_elems());
				std::memset(visited, true, nout);
			}
			else
			{
				for (ade::NElemT i = 0, n = arg.shape_.n_elems(); i < n; ++i)
				{
					if (visited[i])
					{
						acc(out[i], argptr[i]);
					}
					else
					{
						out[i] = argptr[i];
						visited[i] = true;
					}
				}
			}
		}
		else
		{
			ade::CoordT coord;
			if (arg.push_)
			{
				for (ade::NElemT i = 0, n = arg.shape_.n_elems(); i < n; ++i)
				{
					arg.mapper_->forward(coord.begin(),
						ade::coordinate(arg.shape_, i).begin());
					ade::NElemT outidx = ade::index(outshape, coord);
					if (visited[outidx])
					{
						acc(out[outidx], argptr[i]);
					}
					else
					{
						out[outidx] = argptr[i];
						visited[outidx] = true;
					}
				}
			}
			else
			{
				for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
				{
					arg.mapper_->forward(coord.begin(),
						ade::coordinate(outshape, i).begin());
					ade::NElemT inidx = ade::index(arg.shape_, coord);
					if (visited[i])
					{
						acc(out[i], argptr[inidx]);
					}
					else
					{
						out[i] = argptr[inidx];
						visited[i] = true;
					}
				}
			}
		}
	}
	// todo: do something/check unvisited elements
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// sum all elements for all arguments
template <typename T>
void add (T* out, ade::Shape& outshape, DataArgsT<T> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out += val; });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// multiply all elements for all arguments
template <typename T>
void mul (T* out, ade::Shape& outshape, DataArgsT<T> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out *= val; });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the minimum all elements for all arguments
template <typename T>
void min (T* out, ade::Shape& outshape, DataArgsT<T> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out = std::min(out, val); });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
void max (T* out, ade::Shape& outshape, DataArgsT<T> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out = std::max(out, val); });
}

template <typename T>
using  MatrixT = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

#define TO_MAT(ARG)\
Eigen::Map<const MatrixT<T>>(ARG.data_.get(),ARG.shape_.at(1),ARG.shape_.at(0))

template <typename T>
void fast_matmul (T* out, ade::Shape& outshape, DataArg<T> a, DataArg<T> b)
{
	MatrixT<T> mout = TO_MAT(a) * TO_MAT(b);
	std::memcpy(out, mout.data(), sizeof(T) * outshape.n_elems());
}

}

#endif // LLO_OPERATOR_HPP
