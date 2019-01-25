#include "llo/operator.hpp"

#ifdef LLO_OPERATOR_HPP

namespace llo
{

bool is_identity (ade::CoordptrT& coorder)
{
	if (ade::identity == coorder)
	{
		return true;
	}
	bool id = true;
	coorder->access([&id](const ade::MatrixT& m)
	{
		for (uint8_t i = 0; id && i < ade::mat_dim; ++i)
		{
			for (uint8_t j = 0; id && j < ade::mat_dim; ++j)
			{
				id = id && m[i][j] == (i == j);
			}
		}
	});
	return id;
}

EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

#define ABS_INT(INTTYPE)template <>\
void abs<INTTYPE> (TensorT<INTTYPE>& out, DataArg<INTTYPE> in)\
{\
	unary<INTTYPE>(out, in,\
		[](TensorT<INTTYPE>& out, const TensorT<INTTYPE>& src)\
		{ out = src; });\
}

ABS_INT(uint8_t)

ABS_INT(uint16_t)

ABS_INT(uint32_t)

ABS_INT(uint64_t)

#undef ABS_INT

template <>
void neg<uint8_t> (TensorT<uint8_t>& out, DataArg<uint8_t> in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint16_t> (TensorT<uint16_t>& out, DataArg<uint16_t> in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint32_t> (TensorT<uint32_t>& out, DataArg<uint32_t> in)
{
	throw std::bad_function_call();
}

template <>
void neg<uint64_t> (TensorT<uint64_t>& out, DataArg<uint64_t> in)
{
	throw std::bad_function_call();
}

template <>
void rand_uniform<double> (TensorT<double>& out,
	DataArg<double> a, DataArg<double> b)
{
	binary<double>(out, a, b,
		[](TensorT<double>& out, const TensorT<double>& a, const TensorT<double>& b)
		{
			double* outptr = out.data();
			const double* aptr = a.data();
			const double* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				std::uniform_real_distribution<double> dist(aptr[i], bptr[i]);
				outptr[i] = dist(get_engine());
			}
		});
}

template <>
void rand_uniform<float> (TensorT<float>& out,
	DataArg<float> a, DataArg<float> b)
{
	binary<float>(out, a, b,
		[](TensorT<float>& out, const TensorT<float>& a, const TensorT<float>& b)
		{
			float* outptr = out.data();
			const float* aptr = a.data();
			const float* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				std::uniform_real_distribution<float> dist(aptr[i], bptr[i]);
				outptr[i] = dist(get_engine());
			}
		});
}

template <>
void rand_normal<double> (TensorT<double>& out,
	DataArg<double> a, DataArg<double> b)
{
	binary<double>(out, a, b,
		[](TensorT<double>& out, const TensorT<double>& a, const TensorT<double>& b)
		{
			double* outptr = out.data();
			const double* aptr = a.data();
			const double* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				std::normal_distribution<double> dist(aptr[i], bptr[i]);
				outptr[i] = dist(get_engine());
			}
		});
}

template <>
void rand_normal<float> (TensorT<float>& out,
	DataArg<float> a, DataArg<float> b)
{
	binary<float>(out, a, b,
		[](TensorT<float>& out, const TensorT<float>& a, const TensorT<float>& b)
		{
			float* outptr = out.data();
			const float* aptr = a.data();
			const float* bptr = b.data();
			size_t n = get_shape(out).n_elems();
			for (size_t i = 0; i < n; ++i)
			{
				std::normal_distribution<float> dist(aptr[i], bptr[i]);
				outptr[i] = dist(get_engine());
			}
		});
}

}

#endif
