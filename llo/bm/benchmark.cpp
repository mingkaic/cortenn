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


static void BM_MatmulComplex(benchmark::State& state)
{
	std::vector<ade::DimT> alist = {3, 2};
	std::vector<ade::DimT> blist = {4, 3};
	std::vector<ade::DimT> clist = {2, 4};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape(clist);

	llo::VarptrT<int32_t> a(llo::Variable<int32_t>::get(ashape));
	llo::VarptrT<int32_t> b(llo::Variable<int32_t>::get(bshape));
	llo::VarptrT<int32_t> c(llo::Variable<int32_t>::get(cshape));

	ade::TensptrT atens(a);
	ade::TensptrT btens(b);
	ade::TensptrT ctens(c);

	auto d = age::fast_matmul(atens, btens);
	auto e = age::fast_matmul(ctens, d);
	auto f = age::fast_matmul(age::transpose(d), age::transpose(ctens));
	auto dest = age::fast_matmul(e, f);

	ade::TensT ds = llo::multi_derive(dest, {
		a.get(), b.get(), c.get()});

	auto da = ds[0];
	auto db = ds[1];
	auto dc = ds[2];

	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> ddata = random_data(ashape.n_elems(), 1, 100);
		std::vector<double> ddata2 = random_data(bshape.n_elems(), 1, 100);
		std::vector<double> ddata3 = random_data(cshape.n_elems(), 1, 100);
		std::vector<int32_t> data(ddata.begin(), ddata.end());
		std::vector<int32_t> data2(ddata2.begin(), ddata2.end());
		std::vector<int32_t> data3(ddata3.begin(), ddata3.end());
		state.ResumeTiming();
		*a = data;
		*b = data2;
		*c = data3;
		llo::eval<int32_t>(da);
		llo::eval<int32_t>(db);
		llo::eval<int32_t>(dc);
	}
}

BENCHMARK(BM_MatmulComplex);


static void BM_SigmoidMLP(benchmark::State& state)
{
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	llo::VarptrT<double> in(llo::Variable<double>::get(in_shape));
	llo::VarptrT<double> weight0(llo::Variable<double>::get(weight0_shape));
	llo::VarptrT<double> bias0(llo::Variable<double>::get(bias0_shape));
	llo::VarptrT<double> weight1(llo::Variable<double>::get(weight1_shape));
	llo::VarptrT<double> bias1(llo::Variable<double>::get(bias1_shape));
	llo::VarptrT<double> out(llo::Variable<double>::get(out_shape));

	ade::TensptrT intens(in);
	ade::TensptrT weight0tens(weight0);
	ade::TensptrT bias0tens(bias0);
	ade::TensptrT weight1tens(weight1);
	ade::TensptrT bias1tens(bias1);
	ade::TensptrT outtens(out);

	auto layer0 = age::add(age::fast_matmul(intens, weight0tens), age::extend(bias0tens, 1, {3}));
	auto sig0 = age::div(ade::TensptrT(llo::Constant::get(1, ade::Shape({9, 3}))),
		age::add(ade::TensptrT(llo::Constant::get(1, ade::Shape({9, 3}))),
			age::exp(age::neg(layer0))));

	auto layer1 = age::add(age::fast_matmul(sig0, weight1tens), age::extend(bias1tens, 1, {3}));
	auto sig1 = age::div(ade::TensptrT(llo::Constant::get(1, ade::Shape({5, 3}))),
		age::add(ade::TensptrT(llo::Constant::get(1, ade::Shape({5, 3}))),
			age::exp(age::neg(layer1))));

	auto err = age::pow(age::sub(outtens, sig1),
		ade::TensptrT(llo::Constant::get(2, out_shape)));

	ade::TensT ds = llo::multi_derive(err, {
		weight0.get(), bias0.get(), weight1.get(), bias1.get()});

	auto dw0 = ds[0];
	auto db0 = ds[1];
	auto dw1 = ds[2];
	auto db1 = ds[3];
	for (auto _ : state)
	{
		state.PauseTiming();
		std::vector<double> in_data = random_data(in_shape.n_elems(), 0, 1);
		std::vector<double> w0_data = random_data(weight0_shape.n_elems(), 0, 1);
		std::vector<double> b0_data = random_data(bias0_shape.n_elems(), 0, 1);
		std::vector<double> w1_data = random_data(weight1_shape.n_elems(), 0, 1);
		std::vector<double> b1_data = random_data(bias1_shape.n_elems(), 0, 1);
		std::vector<double> out_data = random_data(out_shape.n_elems(), 0, 1);
		state.ResumeTiming();
		*in = in_data;
		*out = out_data;
		*weight0 = w0_data;
		*bias0 = b0_data;
		*weight1 = w1_data;
		*bias1 = b1_data;
		llo::eval<double>(dw0);
		llo::eval<double>(db0);
		llo::eval<double>(dw1);
		llo::eval<double>(db1);
	}
}

BENCHMARK(BM_SigmoidMLP);


BENCHMARK_MAIN();
