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
	/// Raw input data
	const T* data;

	/// Shape info of the raw input
	ade::Shape shape;

	/// Coordinate mapper of input to output
	ade::CoordptrT mapper;

	/// True if data should be pushed input to output (fwd mapper)
	/// False if data should be pulled output from input (bwd mapper)
	bool push;
};

/// Generic unary operation assuming identity mapping
template <typename T>
void unary (T* out, ade::Shape& outshape,
	VecRef<T> in, std::function<T(const T&)> f)
{
	if (in.mapper == ade::identity)
	{
		for (ade::NElemT i = 0, n = in.shape.n_elems(); i < n; ++i)
		{
			out[i] = f(in.data[i]);
		}
	}
	else if (in.push)
	{
		ade::CoordT coord;
		for (ade::NElemT i = 0, n = in.shape.n_elems(); i < n; ++i)
		{
			in.mapper->forward(coord.begin(),
				ade::coordinate(in.shape, i).begin());
			out[ade::index(outshape, coord)] = f(in.data[i]);
		}
	}
	else
	{
		ade::CoordT coord;
		for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
		{
			in.mapper->forward(coord.begin(),
				ade::coordinate(outshape, i).begin());
			out[i] = f(in.data[ade::index(in.shape, coord)]);
		}
	}
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
void abs (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::abs(src); });
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
	unary<T>(out, in.shape, in, [](const T& src) { return -src; });
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
	unary<T>(out, in.shape, in, [](const T& src) { return !src; });
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
void sin (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::sin(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
void cos (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::cos(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
void tan (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::tan(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
void exp (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::exp(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
void log (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::log(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
void sqrt (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::sqrt(src); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
void round (T* out, VecRef<T> in)
{
	unary<T>(out, in.shape, in, [](const T& src) { return std::round(src); });
}

/// Generic binary operation assuming identity mapping
template <typename OUT, typename ATYPE, typename BTYPE>
void binary (OUT* out, ade::Shape& outshape, VecRef<ATYPE> a, VecRef<BTYPE> b,
	std::function<OUT(const ATYPE&,const BTYPE&)> f)
{
	// avoid tmpdata by checking if it's needed
	// tmpdata not needed if neither a nor b are pushing
	if (false == (a.push || b.push))
	{
		ade::CoordT coord;
		ade::CoordT acoord;
		ade::CoordT bcoord;
		for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
		{
			coord = ade::coordinate(outshape, i);
			a.mapper->forward(acoord.begin(), coord.begin());
			b.mapper->forward(bcoord.begin(), coord.begin());
			out[i] = f(
				a.data[ade::index(a.shape, acoord)],
				b.data[ade::index(b.shape, bcoord)]);
		}
	}
	else // a.push || b.push
	{
		std::vector<ATYPE> tmpdata(outshape.n_elems());
		ade::CoordT coord;
		if (a.push)
		{
			for (ade::NElemT i = 0, n = a.shape.n_elems(); i < n; ++i)
			{
				a.mapper->forward(coord.begin(),
					ade::coordinate(a.shape, i).begin());
				tmpdata[ade::index(outshape, coord)] = a.data[i];
			}
		}
		else
		{
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				a.mapper->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				tmpdata[i] = a.data[ade::index(a.shape, coord)];
			}
		}
		if (b.push)
		{
			for (ade::NElemT i = 0, n = b.shape.n_elems(); i < n; ++i)
			{
				b.mapper->forward(coord.begin(),
					ade::coordinate(b.shape, i).begin());
				ade::NElemT outidx = ade::index(outshape, coord);
				out[outidx] = f(tmpdata[outidx], b.data[i]);
			}
		}
		else
		{
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				b.mapper->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				out[i] = f(tmpdata[i], b.data[ade::index(b.shape, coord)]);
			}
		}
	}
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
void pow (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& b, const T& x) { return std::pow(b, x); });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
void sub (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a - b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
void div (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a / b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
void eq (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a == b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply != operator
/// Only accept 2 arguments
template <typename T>
void neq (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a != b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply < operator
/// Only accept 2 arguments
template <typename T>
void lt (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a < b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply > operator
/// Only accept 2 arguments
template <typename T>
void gt (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<T> b)
{
	binary<T,T,T>(out, outshape, a, b,
	[](const T& a, const T& b) { return a > b; });
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::binomial_distribution function
/// Only accept 2 arguments
template <typename T>
void rand_binom (T* out, ade::Shape& outshape, VecRef<T> a, VecRef<double> b)
{
	binary<T,T,double>(out, outshape, a, b,
	[](const T& a, const double& b)
	{
		std::binomial_distribution<T> dist(a, b);
		return dist(get_engine());
	});
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
	binary<T,T,T>(out, outshape, a, b,
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
void rand_normal<double> (double* out,
	ade::Shape& outshape, VecRef<double> a, VecRef<double> b);

template <>
void rand_normal<float> (float* out,
	ade::Shape& outshape, VecRef<float> a, VecRef<float> b);

/// Generic n-nary operation
template <typename T>
void nnary (T* out, ade::Shape& outshape, std::vector<VecRef<T>> args,
	std::function<void(T&, const T&)> acc)
{
	ade::NElemT nout = outshape.n_elems();
	bool visited[nout];
	std::memset(visited, false, nout);
	ade::CoordT coord;
	for (VecRef<T>& arg : args)
	{
		if (arg.push)
		{
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
		else
		{
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				arg.mapper->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				ade::NElemT inidx = ade::index(arg.shape, coord);
				if (visited[i])
				{
					acc(out[i], arg.data[inidx]);
				}
				else
				{
					out[i] = arg.data[inidx];
					visited[i] = true;
				}
			}
		}
	}
	// todo: do something/check unvisited elements
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
