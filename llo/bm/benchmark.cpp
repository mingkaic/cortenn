#include <random>

#include "benchmark/benchmark.h"

#include "llo/llo.hpp"


static std::random_device rnd_device;
static std::mt19937 mersenne_engine(rnd_device());


ade::Shape rand_shape (int n)
{
    std::vector<ade::DimT> slist;
    uint8_t cap = (uint8_t) std::min(256, n);
    for (uint8_t i = 0; i < ade::rank_cap && cap > 1;
        ++i, cap = (uint8_t) std::min(256, n))
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


template <typename T>
static void BM_Abs(benchmark::State& state)
{
    size_t n = state.range(0);
    age::_GENERATED_DTYPE outtype = age::get_type<T>();
	for (auto _ : state)
	{
        state.PauseTiming();
        ade::Shape shape = rand_shape(n);
        std::vector<double> data = random_data(shape.n_elems(), -35, 35);
        llo::VarptrT var = llo::get_variable(data, shape, "var");
        ade::TensptrT out = age::abs(var);
        state.ResumeTiming();
        llo::eval(out, outtype);
	}
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Abs, double)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Abs, float)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Abs, int32_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Abs, int64_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);


template <typename T>
static void BM_Neg(benchmark::State& state)
{
    size_t n = state.range(0);
    age::_GENERATED_DTYPE outtype = age::get_type<T>();
	for (auto _ : state)
	{
        state.PauseTiming();
        ade::Shape shape = rand_shape(n);
        std::vector<double> data = random_data(shape.n_elems(), -35, 35);
        llo::VarptrT var = llo::get_variable(data, shape, "var");
        ade::TensptrT out = age::neg(var);
        state.ResumeTiming();
        llo::eval(out, outtype);
	}
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Neg, double)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Neg, float)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Neg, int32_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Neg, int64_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);


template <typename T>
static void BM_Add(benchmark::State& state)
{
    size_t n = state.range(0);
    age::_GENERATED_DTYPE outtype = age::get_type<T>();
	for (auto _ : state)
	{
        state.PauseTiming();
        ade::Shape shape = rand_shape(n);
        std::vector<double> data = random_data(shape.n_elems(), -35, 35);
        std::vector<double> data2 = random_data(shape.n_elems(), -35, 35);
        llo::VarptrT var = llo::get_variable(data, shape, "var");
        llo::VarptrT var2 = llo::get_variable(data2, shape, "var2");
        ade::TensptrT out = age::add(var, var2);
        state.ResumeTiming();
        llo::eval(out, outtype);
	}
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Add, double)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Add, float)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Add, int32_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Add, int64_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);


template <typename T>
static void BM_Mul(benchmark::State& state)
{
    size_t n = state.range(0);
    age::_GENERATED_DTYPE outtype = age::get_type<T>();
	for (auto _ : state)
	{
        state.PauseTiming();
        ade::Shape shape = rand_shape(n);
        std::vector<double> data = random_data(shape.n_elems(), -35, 35);
        std::vector<double> data2 = random_data(shape.n_elems(), -35, 35);
        llo::VarptrT var = llo::get_variable(data, shape, "var");
        llo::VarptrT var2 = llo::get_variable(data2, shape, "var2");
        ade::TensptrT out = age::mul(var, var2);
        state.ResumeTiming();
        llo::eval(out, outtype);
	}
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Mul, double)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Mul, float)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Mul, int32_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE(BM_Mul, int64_t)
    ->Range(64, 2048)
    ->Complexity(benchmark::oN);


template <typename T>
static void BM_Matmul(benchmark::State& state)
{
    size_t n = state.range(0);
    age::_GENERATED_DTYPE outtype = age::get_type<T>();
	for (auto _ : state)
	{
        state.PauseTiming();
        std::uniform_int_distribution<> distc(1, std::min(256ul, n - 1));
        ade::DimT common_dim = distc(mersenne_engine);
        int remaining = n / common_dim;
        std::uniform_int_distribution<> distsides(1, std::min(256, remaining));
        ade::DimT left_dim = distsides(mersenne_engine);
        ade::DimT right_dim = distsides(mersenne_engine);
        ade::Shape leftshape({common_dim, left_dim});
        ade::Shape rightshape({right_dim, common_dim});
        std::vector<double> data = random_data(leftshape.n_elems(), -35, 35);
        std::vector<double> data2 = random_data(rightshape.n_elems(), -35, 35);
        llo::VarptrT var = llo::get_variable(data, leftshape, "var");
        llo::VarptrT var2 = llo::get_variable(data2, rightshape, "var2");
        ade::TensptrT out = age::matmul(var, var2);
        state.ResumeTiming();
        llo::eval(out, outtype);
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
