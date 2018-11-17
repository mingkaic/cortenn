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

#include "ade/coord.hpp"

#ifndef LLO_OPERATOR_HPP
#define LLO_OPERATOR_HPP

namespace llo
{

/// RNG engine used
using EngineT = std::default_random_engine;

/// Return global random generator
EngineT& get_engine (void);

/// Tensor data wrapper using raw pointer and data size
/// Avoid using std constainers in case of unintentional deep copies
template <typename T>
struct VecRef
{
	const T* data;

	ade::Shape shape;

	ade::CoordPtrT mapper;
};

/// Generic unary operation assuming identity mapping (bijective)
template <typename T>
void unary (T* out, VecRef<T> in, std::function<T(const T&)> f)
{
	ade::NElemT n = in.shape.n_elems();
	for (ade::NElemT i = 0; i < n; ++i)
	{
		out[i] = f(in.data[i]);
	}
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
void abs (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::abs(src); });
}

template <>
void abs<uint8_t> (uint8_t* out, VecRef<uint8_t> in);

template <>
void abs<uint16_t> (uint16_t* out, VecRef<uint16_t> in);

template <>
void abs<uint32_t> (uint32_t* out, VecRef<uint32_t> in);

template <>
void abs<uint64_t> (uint64_t* out, VecRef<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
void neg (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return -src; });
}

template <>
void neg<uint8_t> (uint8_t* out, VecRef<uint8_t> in);

template <>
void neg<uint16_t> (uint16_t* out, VecRef<uint16_t> in);

template <>
void neg<uint32_t> (uint32_t* out, VecRef<uint32_t> in);

template <>
void neg<uint64_t> (uint64_t* out, VecRef<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take bitwise nots of inputs
template <typename T>
void bit_not (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return !src; });
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
void sin (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::sin(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
void cos (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::cos(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
void tan (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::tan(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
void exp (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::exp(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
void log (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::log(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
void sqrt (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::sqrt(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
void round (T* out, VecRef<T> in)
{
	unary<T>(out, in, [](const T& src) { return std::round(src); });
}

/// Generic binary operation assuming identity mapping (bijective)
template <typename T>
void binary (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b,
	std::function<T(const T&,const T&)> f)
{
	ade::NElemT n = outshape.n_elems();
	ade::CoordT acoord;
	ade::CoordT bcoord;
	for (ade::NElemT i = 0; i < n; ++i)
	{
		a.mapper->backward(acoord.begin(),
			ade::coordinate(outshape, i).begin());
		b.mapper->backward(bcoord.begin(),
			ade::coordinate(outshape, i).begin());
		out[i] = f(a.data[ade::index(a.shape, acoord)],
			b.data[ade::index(b.shape, bcoord)]);
	}
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
void pow (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& b, const T& x) { return std::pow(b, x); });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
void sub (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& a, const T& b) { return a - b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
void div (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& a, const T& b) { return a / b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
void eq (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& a, const T& b) { return a == b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply != operator
/// Only accept 2 arguments
template <typename T>
void neq (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& a, const T& b) { return a != b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply < operator
/// Only accept 2 arguments
template <typename T>
void lt (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& a, const T& b) { return a < b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply > operator
/// Only accept 2 arguments
template <typename T>
void gt (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T>(out, outshape, a, b,
		[](const T& a, const T& b) { return a > b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::binomial_distribution function
/// Only accept 2 arguments
template <typename T>
void rand_binom (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<double> b)
{
	ade::NElemT n = outshape.n_elems();
	ade::CoordT acoord;
	ade::CoordT bcoord;
	for (ade::NElemT i = 0; i < n; ++i)
	{
		a.mapper->backward(acoord.begin(),
			ade::coordinate(outshape, i).begin());
		b.mapper->backward(bcoord.begin(),
			ade::coordinate(outshape, i).begin());
		std::binomial_distribution<T> dist(
			a.data[ade::index(a.shape, acoord)],
			b.data[ade::index(b.shape, bcoord)]);
		out[i] = dist(get_engine());
	}
}

template <>
void rand_binom<double> (double* out,
	ade::Shape& outshape, VecRef<double> a, VecRef<double> b);

template <>
void rand_binom<float> (float* out,
	ade::Shape& outshape, VecRef<float> a, VecRef<double> b);

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T>
void rand_uniform (T* out,
	ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
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
	ade::Shape& outshape, VecRef<double> a, VecRef<double> b);

template <>
void rand_uniform<float> (float* out,
	ade::Shape& outshape, VecRef<float> a, VecRef<float> b);

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::normal_distribution function
/// Only accept 2 arguments
template <typename T>
void rand_normal (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<float> (float* out,
	ade::Shape& outshape, VecRef<float> a, VecRef<float> b);

template <>
void rand_normal<double> (double* out,
	ade::Shape& outshape, VecRef<double> a, VecRef<double> b);

/// Generic n-nary operation (potentially surjective)
template <typename T>
void nnary (T* out, ade::Shape& outshape, std::vector<VecRef<T>> args,
	std::function<void(T&, const T&)> acc)
{
	ade::NElemT nout = outshape.n_elems();
	bool visited[nout];
	std::memset(visited, false, nout);
	ade::CoordT coord;
	size_t nargs = args.size();
	if (nargs == 1 && nout > args[0].shape.n_elems()) // resolve extensions
	{
		VecRef<T>& arg = args[0];
		for (ade::NElemT outidx = 0; outidx < nout; ++outidx)
		{
			arg.mapper->backward(coord.begin(),
				ade::coordinate(outshape, outidx).begin());
			ade::NElemT i = ade::index(arg.shape, coord);
			out[outidx] = arg.data[i];
		}
	}
	else
	{
		for (size_t i = 0; i < nargs; ++i)
		{
			VecRef<T>& arg = args[i];
			for (ade::NElemT i = 0, n = arg.shape.n_elems(); i < n; ++i)
			{
				arg.mapper->forward(coord.begin(),
					ade::coordinate(arg.shape, i).begin());
				ade::NElemT outidx = ade::index(outshape, coord);
				if (visited[outidx])
				{
					acc(out[outidx], arg.data[i]);
				}
				else
				{
					out[outidx] = arg.data[i];
					visited[outidx] = true;
				}
			}
		}
		// todo: do something/check non-visited elements
	}
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// sum all elements for all arguments
template <typename T>
void add (T* out, ade::Shape& outshape, std::vector<VecRef<T>> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out += val; });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// multiply all elements for all arguments
template <typename T>
void mul (T* out, ade::Shape& outshape, std::vector<VecRef<T>> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out *= val; });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the minimum all elements for all arguments
template <typename T>
void min (T* out, ade::Shape& outshape, std::vector<VecRef<T>> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out = std::min(out, val); });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
void max (T* out, ade::Shape& outshape, std::vector<VecRef<T>> args)
{
	nnary<T>(out, outshape, args,
		[](T& out, const T& val) { out = std::max(out, val); });
}

}

#endif // LLO_OPERATOR_HPP
