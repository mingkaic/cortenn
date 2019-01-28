#include <random>

#include "benchmark/benchmark.h"

#include "llo/llo.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


ade::Shape rand_shape (int n)
{
	std::vector<ade::DimT> slist;
	uint8_t cap = (uint8_t) std::min(255, n);
	for (uint8_t i = 0; i < ade::rank_cap && cap > 1;
		++i, cap = (uint8_t) std::min(255, n))
	{
		std::uniform_int_distribution<> dist(1, cap);
		uint8_t c = dist(mersenne_engine);
		n /= c;
		slist.push_back(c);
	}
	return ade::Shape(slist);
}


static std::vector<double> random_data (size_t n, double lower, double upper)
{
	std::vector<double> out(n);
	std::uniform_real_distribution<double> dist(lower, upper);
	std::generate(out.begin(), out.end(),
		[&dist]() { return dist(mersenne_engine); });
	return out;
}


#define DEFN_BENCHMARK(NAME, FUNC, DEFN)\
DEFN(NAME, FUNC)\
BENCHMARK_TEMPLATE(NAME, double)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, float)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, int32_t)->Range(64, 2048)\
	->Complexity(benchmark::oN);\
BENCHMARK_TEMPLATE(NAME, int64_t)->Range(64, 2048)\
	->Complexity(benchmark::oN);


#define DEFN_UNARY(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		ade::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), -35, 35);\
		llo::VarptrT<double> var(llo::Variable<double>::get(data, shape, "var"));\
		ade::TensptrT out = FUNC(var);\
		state.ResumeTiming();\
		llo::eval<T>(out);\
	}\
	state.SetComplexityN(state.range(0));\
}


#define DEFN_UNARY_POS(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		ade::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), 0, 35);\
		llo::VarptrT<double> var(llo::Variable<double>::get(data, shape, "var"));\
		ade::TensptrT out = FUNC(var);\
		state.ResumeTiming();\
		llo::eval<T>(out);\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Abs, age::abs, DEFN_UNARY)


DEFN_BENCHMARK(BM_Neg, age::neg, DEFN_UNARY)


DEFN_BENCHMARK(BM_Sin, age::sin, DEFN_UNARY)


DEFN_BENCHMARK(BM_Cos, age::cos, DEFN_UNARY)


DEFN_BENCHMARK(BM_Tan, age::tan, DEFN_UNARY)


DEFN_BENCHMARK(BM_Exp, age::exp, DEFN_UNARY)


DEFN_BENCHMARK(BM_Log, age::log, DEFN_UNARY_POS)


DEFN_BENCHMARK(BM_Sqrt, age::sqrt, DEFN_UNARY_POS)


DEFN_BENCHMARK(BM_Round, age::round, DEFN_UNARY)


#define DEFN_BINARY(NAME, FUNC)\
template <typename T>\
static void NAME(benchmark::State& state)\
{\
	size_t n = state.range(0);\
	for (auto _ : state)\
	{\
		state.PauseTiming();\
		ade::Shape shape = rand_shape(n);\
		std::vector<double> data = random_data(shape.n_elems(), 1, 4);\
		std::vector<double> data2 = random_data(shape.n_elems(), 1, 4);\
		llo::VarptrT<double> var(llo::Variable<double>::get(data, shape, "var"));\
		llo::VarptrT<double> var2(llo::Variable<double>::get(data2, shape, "var2"));\
		ade::TensptrT out = FUNC(var, var2);\
		state.ResumeTiming();\
		llo::eval<T>(out);\
	}\
	state.SetComplexityN(state.range(0));\
}


DEFN_BENCHMARK(BM_Pow, age::pow, DEFN_BINARY)


DEFN_BENCHMARK(BM_Add, age::add, DEFN_BINARY)


DEFN_BENCHMARK(BM_Sub, age::sub, DEFN_BINARY)


DEFN_BENCHMARK(BM_Mul, age::mul, DEFN_BINARY)


DEFN_BENCHMARK(BM_Div, age::div, DEFN_BINARY)


DEFN_BENCHMARK(BM_Eq, age::eq, DEFN_BINARY)


DEFN_BENCHMARK(BM_Ne, age::neq, DEFN_BINARY)


DEFN_BENCHMARK(BM_Lt, age::lt, DEFN_BINARY)


DEFN_BENCHMARK(BM_Gt, age::gt, DEFN_BINARY)


template <typename T>
static void BM_Matmul(benchmark::State& state)
{
	size_t n = state.range(0);
	for (auto _ : state)
	{
		state.PauseTiming();
		std::uniform_int_distribution<ade::DimT> distc(9, std::min(255ul, n - 1));
		ade::DimT common_dim = distc(mersenne_engine);
		int remaining = (double) n / common_dim;
		std::uniform_int_distribution<> distsides(1, std::min(255, remaining));
		ade::DimT left_dim = distsides(mersenne_engine);
		ade::DimT right_dim = distsides(mersenne_engine);
		ade::Shape leftshape({common_dim, left_dim});
		ade::Shape rightshape({right_dim, common_dim});
		std::vector<double> data = random_data(leftshape.n_elems(), -35, 35);
		std::vector<double> data2 = random_data(rightshape.n_elems(), -35, 35);
		llo::VarptrT<double> var(llo::Variable<double>::get(data, leftshape, "var"));
		llo::VarptrT<double> var2(llo::Variable<double>::get(data2, rightshape, "var2"));
		ade::TensptrT out = age::fast_matmul(var, var2);
		state.ResumeTiming();
		llo::eval<T>(out);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Matmul, double)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Matmul, float)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Matmul, int32_t)
	->Range(64, 2048)
	->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Matmul, int64_t)
	->Range(64, 2048)
	->Complexity(benchmark::oN);


BENCHMARK_MAIN();
