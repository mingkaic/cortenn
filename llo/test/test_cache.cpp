
#ifndef DISABLE_CACHE_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/generated/api.hpp"

#include "llo/cache.hpp"
#include "llo/constant.hpp"
#include "llo/variable.hpp"
#include "llo/eval.hpp"


TEST(CACHE, CacheLocations)
{
	std::vector<ade::DimT> slist = {4, 3, 2};
	ade::Shape shape(slist);

	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 56, 3,
		7, 22, 56, 50, 19, 13, 12, 10, 31, 40, 24, 11
	};
	ade::TensptrT src(llo::Variable<double>::get(data, shape));
	ade::TensptrT src2(llo::Variable<double>::get(data, shape));
	ade::TensptrT cst(llo::Constant::get(1, shape));

	auto nocache_f = age::sum({src, cst});

	auto cached_f = age::prod({nocache_f, src2});

	auto nocached_u = age::abs(cached_f);

	llo::CacheSpace<double> caches({cached_f});

	EXPECT_FALSE(caches.has_value(
		static_cast<ade::iFunctor*>(nocache_f.get())));
	EXPECT_TRUE(caches.has_value(
		static_cast<ade::iFunctor*>(cached_f.get())));
	EXPECT_FALSE(caches.has_value(
		static_cast<ade::iFunctor*>(nocached_u.get())));
}


TEST(CACHE, CacheMark)
{
	std::vector<ade::DimT> slist = {4, 3, 2};
	ade::Shape shape(slist);

	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 56, 3,
		7, 22, 56, 50, 19, 13, 12, 10, 31, 40, 24, 11
	};
	ade::TensptrT src(llo::Variable<double>::get(data, shape));
	ade::TensptrT src2(llo::Variable<double>::get(data, shape));
	auto v = static_cast<llo::iVariable*>(src.get());
	auto v2 = static_cast<llo::iVariable*>(src2.get());
	ade::TensptrT cst(llo::Constant::get(1, shape));

	auto nocache_f = age::sum({src, cst});
	auto cached_f = age::prod({nocache_f, src2});
	auto nocached_u = age::abs(cached_f);
	auto f = static_cast<ade::iFunctor*>(cached_f.get());

	llo::CacheSpace<double> caches({cached_f});
	EXPECT_EQ(nullptr, caches.get(f));

	auto out = llo::eval(nocached_u, &caches);
	std::vector<double> got_result;
	{
		auto gotshape = out->dimensions();
		ASSERT_ARREQ(slist, gotshape);
		size_t n = shape.n_elems();
		double* gotptr = out->data();
		got_result = std::vector<double>(gotptr, gotptr + n);
	}

	auto tens = caches.get(f);
	EXPECT_NE(nullptr, tens);

	{
		auto gotshape = tens->dimensions();
		ASSERT_ARREQ(slist, gotshape);
		size_t n = shape.n_elems();
		double* gotptr = (double*) tens->data();
		std::vector<double> cached_result(gotptr, gotptr + n);
		EXPECT_ARREQ(got_result, cached_result);
	}

	caches.mark_update({v, v2});
	EXPECT_EQ(nullptr, caches.get(f));
}


#endif // DISABLE_CACHE_TEST
