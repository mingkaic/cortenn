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
inline void unary (TensorT<T>& out, DataArg<T> in,
	std::function<void(TensorT<T>&,const TensorT<T>&)> f)
{
	if (is_identity(in.mapper_))
	{
		f(out, in.data_);
	}
	else
	{
		ade::Shape outshape = get_shape(out);
		// else map coordinates then apply f
		TensorT<T> temp = get_tensor<T>(nullptr, outshape);
		ade::Shape inshape = get_shape(in.data_);

		T* tempptr = temp.data();
		const T* inptr = in.data_.data();
		if (in.push_)
		{
			ade::CoordT coord;
			for (ade::NElemT i = 0, n = inshape.n_elems(); i < n; ++i)
			{
				in.mapper_->forward(coord.begin(),
					ade::coordinate(inshape, i).begin());
				tempptr[ade::index(outshape, coord)] = inptr[i];
			}
		}
		else
		{
			ade::CoordT coord;
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				in.mapper_->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				tempptr[i] = inptr[ade::index(inshape, coord)];
			}
		}
		f(out, temp);
	}
}

/// Given reference to output array, and input vector ref,
/// make output elements take absolute value of inputs
template <typename T>
void abs (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{ out = src.abs(); });
}

template <>
void abs<uint8_t> (TensorT<uint8_t>& out, DataArg<uint8_t> in);

template <>
void abs<uint16_t> (TensorT<uint16_t>& out, DataArg<uint16_t> in);

template <>
void abs<uint32_t> (TensorT<uint32_t>& out, DataArg<uint32_t> in);

template <>
void abs<uint64_t> (TensorT<uint64_t>& out, DataArg<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take negatives of inputs
template <typename T>
void neg (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{ out = -src; });
}

template <>
void neg<uint8_t> (TensorT<uint8_t>& out, DataArg<uint8_t> in);

template <>
void neg<uint16_t> (TensorT<uint16_t>& out, DataArg<uint16_t> in);

template <>
void neg<uint32_t> (TensorT<uint32_t>& out, DataArg<uint32_t> in);

template <>
void neg<uint64_t> (TensorT<uint64_t>& out, DataArg<uint64_t> in);

/// Given reference to output array, and input vector ref,
/// make output elements take bitwise nots of inputs
template <typename T>
void bit_not (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{
			T* outptr = out.data();
			const T* inptr = src.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = !inptr[i];
			}
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take sine of inputs
template <typename T>
void sin (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{
			T* outptr = out.data();
			const T* inptr = src.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = std::sin(inptr[i]);
			}
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take cosine of inputs
template <typename T>
void cos (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{
			T* outptr = out.data();
			const T* inptr = src.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = std::cos(inptr[i]);
			}
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take tangent of inputs
template <typename T>
void tan (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{
			T* outptr = out.data();
			const T* inptr = src.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = std::tan(inptr[i]);
			}
		});
}

/// Given reference to output array, and input vector ref,
/// make output elements take exponent of inputs
template <typename T>
void exp (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{ out = src.exp(); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take natural log of inputs
template <typename T>
void log (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{ out = src.log(); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take square root of inputs
template <typename T>
void sqrt (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{ out = src.sqrt(); });
}

/// Given reference to output array, and input vector ref,
/// make output elements take rounded values of inputs
template <typename T>
void round (TensorT<T>& out, DataArg<T> in)
{
	unary<T>(out, in,
		[](TensorT<T>& out, const TensorT<T>& src)
		{ out = src.round(); });
}

/// Generic binary operation assuming identity mapping
template <typename T>
inline void binary (TensorT<T>& out, DataArg<T> a, DataArg<T> b,
	std::function<void(TensorT<T>&,const TensorT<T>&,const TensorT<T>&)> f)
{
	ade::Shape outshape = get_shape(out);
	TensorT<T>* lhs = nullptr;
	TensorT<T>* rhs = nullptr;

	if (is_identity(a.mapper_))
	{
		lhs = &a.data_;
	}
	else // map coordinates
	{
		lhs = new TensorT<T>(out.dimensions());
		lhs->setZero();
		ade::Shape inshape = get_shape(a.data_);

		T* ptr = lhs->data();
		const T* inptr = a.data_.data();
		if (a.push_)
		{
			ade::CoordT coord;
			for (ade::NElemT i = 0, n = inshape.n_elems(); i < n; ++i)
			{
				a.mapper_->forward(coord.begin(),
					ade::coordinate(inshape, i).begin());
				ptr[ade::index(outshape, coord)] = inptr[i];
			}
		}
		else
		{
			ade::CoordT coord;
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				a.mapper_->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				ptr[i] = inptr[ade::index(inshape, coord)];
			}
		}
	}

	if (is_identity(b.mapper_))
	{
		rhs = &b.data_;
	}
	else // map coordinates
	{
		rhs = new TensorT<T>(out.dimensions());
		rhs->setZero();
		ade::Shape inshape = get_shape(b.data_);

		T* ptr = rhs->data();
		const T* inptr = b.data_.data();
		if (b.push_)
		{
			ade::CoordT coord;
			for (ade::NElemT i = 0, n = inshape.n_elems(); i < n; ++i)
			{
				b.mapper_->forward(coord.begin(),
					ade::coordinate(inshape, i).begin());
				ptr[ade::index(outshape, coord)] = inptr[i];
			}
		}
		else
		{
			ade::CoordT coord;
			for (ade::NElemT i = 0, n = outshape.n_elems(); i < n; ++i)
			{
				b.mapper_->forward(coord.begin(),
					ade::coordinate(outshape, i).begin());
				ptr[i] = inptr[ade::index(inshape, coord)];
			}
		}
	}

	f(out, *lhs, *rhs);
	if (lhs != &a.data_)
	{
		delete lhs;
	}
	if (rhs != &b.data_)
	{
		delete rhs;
	}
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::pow operator
/// Only accept 2 arguments
template <typename T>
void pow (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& b, const TensorT<T>& x)
		{
			T* outptr = out.data();
			const T* bptr = b.data();
			const T* xptr = x.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = std::pow(bptr[i], xptr[i]);
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index subtract
/// Only accept 2 arguments
template <typename T>
void sub (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			out = a - b;
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index divide
/// Only accept 2 arguments
template <typename T>
void div (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			out = a / b;
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply == operator
/// Only accept 2 arguments
template <typename T>
void eq (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			T* outptr = out.data();
			const T* aptr = a.data();
			const T* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = aptr[i] == bptr[i];
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply != operator
/// Only accept 2 arguments
template <typename T>
void neq (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			T* outptr = out.data();
			const T* aptr = a.data();
			const T* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = aptr[i] != bptr[i];
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply < operator
/// Only accept 2 arguments
template <typename T>
void lt (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			T* outptr = out.data();
			const T* aptr = a.data();
			const T* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = aptr[i] < bptr[i];
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply > operator
/// Only accept 2 arguments
template <typename T>
void gt (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	return binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			T* outptr = out.data();
			const T* aptr = a.data();
			const T* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = aptr[i] > bptr[i];
			}
		});
}

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::uniform_distributon function
/// Only accept 2 arguments
template <typename T>
void rand_uniform (TensorT<T>& out,
	DataArg<T> a, DataArg<T> b)
{
	binary<T>(out, a, b,
		[](TensorT<T>& out, const TensorT<T>& a, const TensorT<T>& b)
		{
			T* outptr = out.data();
			const T* aptr = a.data();
			const T* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				std::uniform_int_distribution<T> dist(aptr[i], bptr[i]);
				outptr[i] = dist(get_engine());
			}
		});
}

template <>
void rand_uniform<double> (TensorT<double>& out,
	DataArg<double> a, DataArg<double> b);

template <>
void rand_uniform<float> (TensorT<float>& out,
	DataArg<float> a, DataArg<float> b);

/// Given arguments a, and b, for every pair of mapped elements sharing the
/// same index apply std::normal_distribution function
/// Only accept 2 arguments
template <typename T>
void rand_normal (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	throw std::bad_function_call();
}

template <>
void rand_normal<double> (TensorT<double>& out,
	DataArg<double> a, DataArg<double> b);

template <>
void rand_normal<float> (TensorT<float>& out,
	DataArg<float> a, DataArg<float> b);

/// Generic n-nary operation
template <typename T>
inline void nnary (TensorT<T>& out, DataArgsT<T> args,
	std::function<void(T&,const T&)> acc,
	std::function<void(TensorT<T>&,const TensorT<T>&)> tensacc)
{
	ade::Shape outshape = get_shape(out);
	ade::NElemT nout = outshape.n_elems();
	bool visited[nout];
	std::memset(visited, false, nout);

	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		auto& arg = args[i];
		if (is_identity(arg.mapper_))
		{
			if (i == 0)
			{
				out = arg.data_;
				std::memset(visited, true, nout);
			}
			else
			{
				tensacc(out, arg.data_);
			}
		}
		else // map coordinates
		{
			ade::CoordT coord;
			ade::Shape inshape = get_shape(arg.data_);

			T* ptr = out.data();
			const T* argptr = arg.data_.data();
			if (arg.push_)
			{
				for (ade::NElemT i = 0, n = inshape.n_elems(); i < n; ++i)
				{
					arg.mapper_->forward(coord.begin(),
						ade::coordinate(inshape, i).begin());
					ade::NElemT outidx = ade::index(outshape, coord);
					if (visited[outidx])
					{
						acc(ptr[outidx], argptr[i]);
					}
					else
					{
						ptr[outidx] = argptr[i];
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
					ade::NElemT inidx = ade::index(inshape, coord);
					if (visited[i])
					{
						acc(ptr[i], argptr[inidx]);
					}
					else
					{
						ptr[i] = argptr[inidx];
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
void add (TensorT<T>& out, DataArgsT<T> args)
{
	nnary<T>(out, args,
		[](T& out, const T& val) { out += val; },
		[](TensorT<T>& out, const TensorT<T>& val) { out += val; });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// multiply all elements for all arguments
template <typename T>
void mul (TensorT<T>& out, DataArgsT<T> args)
{
	nnary<T>(out, args,
		[](T& out, const T& val) { out *= val; },
		[](TensorT<T>& out, const TensorT<T>& val) { out *= val; });
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the minimum all elements for all arguments
template <typename T>
void min (TensorT<T>& out, DataArgsT<T> args)
{
	nnary<T>(out, args,
		[](T& out, const T& val) { out = std::min(out, val); },
		[](TensorT<T>& out, const TensorT<T>& val)
		{
			T* outptr = out.data();
			const T* valptr = val.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = std::min(outptr[i], valptr[i]);
			}
		});
}

/// Given arguments, for every mapped index i in range [0:max_nelems],
/// take the maximum all elements for all arguments
template <typename T>
void max (TensorT<T>& out, DataArgsT<T> args)
{
	nnary<T>(out, args,
		[](T& out, const T& val) { out = std::max(out, val); },
		[](TensorT<T>& out, const TensorT<T>& val)
		{
			T* outptr = out.data();
			const T* valptr = val.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				outptr[i] = std::max(outptr[i], valptr[i]);
			}
		});
}

template <typename T>
using  MatrixT = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

#define TO_MAT(ARG)\
Eigen::Map<const MatrixT<T>>(ARG.data_.data(),\
	ARG.data_.dimension(1),ARG.data_.dimension(0))

template <typename T>
void fast_matmul (TensorT<T>& out, DataArg<T> a, DataArg<T> b)
{
	MatrixT<T> mout = TO_MAT(a) * TO_MAT(b);
	out = Eigen::TensorMap<TensorT<T>>(mout.data(), {
		mout.cols(), mout.rows(), 1, 1, 1, 1, 1, 1});
}

}

#endif // LLO_OPERATOR_HPP
