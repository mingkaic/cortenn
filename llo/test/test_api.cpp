
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"
#include "retroc/rand.hpp"

#include "bwd/grader.hpp"

#include "llo/generated/api.hpp"

#include "llo/eval.hpp"
#include "llo/zprune.hpp"


using UnaryDblF = std::function<double(double)>;

using UnaryOpF = std::function<ade::TensptrT(ade::TensptrT&)>;

using BinaryOpF = std::function<ade::TensptrT(ade::TensptrT&,ade::TensptrT&)>;

template <typename T>
using BinaryFwdF = std::function<T(T,T)>;

template <typename T>
using BinaryBwdF = std::function<T(T,T,T,T)>;

using MatVecT = std::vector<std::vector<int32_t>>;

static const retro::Range<double> default_range = {-9876, 9876};

const int FREIVALD_N = 10;


struct API : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


MatVecT create_2d (llo::GenericData& data)
{
	int32_t* ptr = (int32_t*) data.data_.get();
	std::vector<ade::DimT> dims(data.shape_.begin(), data.shape_.end());
	ade::DimT C = dims[0];
	ade::DimT R = dims[1];
	MatVecT res;

 	for (size_t y = 0; y < R; y++)
	{
		res.push_back(std::vector<signed>(C, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			res[y][x] = ptr[x + y * C];
		}
	}
	return res;
}


bool freivald (MatVecT a, MatVecT b, MatVecT c)
{
	uint8_t cdim = b.size();
	uint8_t bdim = b[0].size();
	uint8_t adim = a.size();
	// a has shape [cdim, adim]
	// b has shape [bdim, cdim]
	// c has shape [bdim, adim]
	// probability of false positive = 1/2^n
	// Pr(fp) = 0.1% ~~> n = 10
	for (int i = 0; i < FREIVALD_N; i++)
	{
		// generate r of len b[0].size() or c[0].size()
		std::vector<int32_t> r = retro::get_vec<int32_t>(bdim, {0, 1});

		// p = matmul(a, matmul(b, r)) - matmul(c, r)
		std::vector<int32_t> br; // matmul(b, r)
		for (size_t y = 0; y < cdim; y++)
		{
			int32_t bri = 0;
			for (size_t x = 0; x < bdim; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<int32_t> cr; // matmul(c, r)
		for (size_t y = 0; y < adim; y++)
		{
			int32_t cri = 0;
			for (size_t x = 0; x < bdim; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<int32_t> p;
		for (size_t y = 0; y < adim; y++)
		{
			int32_t ari = 0;
			for (size_t x = 0, m = a[y].size(); x < m; x++)
			{
				ari += a[y][x] * br[x];
			}
			p.push_back(ari);
		}
		for (size_t j = 0; j < adim; j++)
		{
			p[j] -= cr[j];
		}

		// if p != 0 -> return false
		if (!std::all_of(p.begin(), p.end(),
			[](int32_t d) { return d == 0; }))
		{
			return false;
		}
	}
	return true;
}


static void unary_generic (simple::SessionT& sess,
	retro::Range<double> range, UnaryOpF op,
	std::function<void(llo::GenericData&,ade::Shape&,std::vector<double>&)> verify,
	std::function<void(double*,std::vector<double>&)> bwverify)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);

	ade::TensptrT src = llo::get_variable<double>(data, shape);
	ade::TensptrT dest = op(src);

	llo::GenericData out = llo::eval(dest, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, out.dtype_);
	verify(out, shape, data);

	ade::TensptrT gsrc = llo::derive(dest, src.get());

	llo::GenericData gout = llo::eval(gsrc, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout.dtype_);
	std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
	ASSERT_ARREQ(slist, gotshape);
	double* goptr = (double*) gout.data_.get();
	bwverify(goptr, data);
}


static void unary_elementary (simple::SessionT& sess,
	retro::Range<double> range, UnaryOpF op,
	UnaryDblF fwd, UnaryDblF bwd, bool save_grad = true)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, range);

	ade::TensptrT src = llo::get_variable<double>(data, shape);
	ade::TensptrT dest = op(src);

	llo::GenericData out = llo::eval(dest, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, out.dtype_);
	{
		std::vector<ade::DimT> gotshape(out.shape_.begin(), out.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* optr = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>(optr, optr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(fwd(data[i]), optr[i]);
		}
	});

	ade::TensptrT gsrc = llo::derive(dest, src.get());

	llo::GenericData gout = llo::eval(gsrc, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	auto verify = [&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i]), goptr[i]);
		}
	};
	if (false == save_grad)
	{
		verify();
	}
	else
	{
		double_verify(sess, "gout", std::vector<double>(goptr, goptr + n), verify);
	}
}


static void binary_elementary (simple::SessionT& sess,
	retro::Range<double> range, BinaryOpF op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, range);
	std::vector<double> data2 = sess->get_double("data2", n, range);

	ade::TensptrT src = llo::get_variable<double>(data, shape);
	ade::TensptrT src2 = llo::get_variable<double>(data2, shape);
	ade::TensptrT dest = op(src, src2);

	llo::GenericData out = llo::eval(dest, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, out.dtype_);
	{
		std::vector<ade::DimT> gotshape(out.shape_.begin(), out.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* optr = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>(optr, optr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(fwd(data[i], data2[i]), optr[i]);
		}
	});

	ade::TensptrT dest2 = op(src, src);
	ade::TensptrT gsame = llo::derive(dest2, src.get());
	llo::GenericData gout = llo::eval(gsame, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	double_verify(sess, "gout", std::vector<double>(goptr, goptr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1.0, 1.0), goptr[i]);
		}
	});

	ade::TensptrT gleft = llo::derive(dest, src.get());
	llo::GenericData gout_left = llo::eval(gleft, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout_left.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout_left.shape_.begin(), gout_left.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr2 = (double*) gout_left.data_.get();
	double_verify(sess, "gout_left", std::vector<double>(goptr2, goptr2 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1.0, 0.0), goptr2[i]);
		}
	});

	ade::TensptrT gright = llo::derive(dest, src2.get());
	llo::GenericData gout_right = llo::eval(gright, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout_right.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout_right.shape_.begin(), gout_right.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr3 = (double*) gout_right.data_.get();
	double_verify(sess, "gout_right", std::vector<double>(goptr3, goptr3 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0.0, 1.0), goptr3[i]);
		}
	});
}


static void binary_elementary_int (simple::SessionT& sess,
	retro::Range<int32_t> range, BinaryOpF op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<int32_t> data = sess->get_int("data", n, range);
	std::vector<int32_t> data2 = sess->get_int("data2", n, range);

	ade::TensptrT src = llo::get_variable<int32_t>(data, shape);
	ade::TensptrT src2 = llo::get_variable<int32_t>(data2, shape);
	ade::TensptrT dest = op(src, src2);

	llo::GenericData out = llo::eval(dest, age::INT32);
	ASSERT_EQ(age::INT32, out.dtype_);
	{
		std::vector<ade::DimT> gotshape(out.shape_.begin(), out.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	int32_t* optr = (int32_t*) out.data_.get();
	int_verify(sess, "out", std::vector<int32_t>(optr, optr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
		}
	});

	ade::TensptrT dest2 = op(src, src);
	ade::TensptrT gsame = llo::derive(dest2, src.get());
	llo::GenericData gout = llo::eval(gsame, age::INT32);
	ASSERT_EQ(age::INT32, gout.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	int32_t* goptr = (int32_t*) gout.data_.get();
	int_verify(sess, "gout", std::vector<int32_t>(goptr, goptr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(bwd(data[i], data[i], 1, 1), goptr[i]);
		}
	});

	ade::TensptrT gleft = llo::derive(dest, src.get());
	llo::GenericData gout_left = llo::eval(gleft, age::INT32);
	ASSERT_EQ(age::INT32, gout_left.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout_left.shape_.begin(), gout_left.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	int32_t* goptr2 = (int32_t*) gout_left.data_.get();
	int_verify(sess, "gout_left", std::vector<int32_t>(goptr2, goptr2 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(bwd(data[i], data2[i], 1, 0), goptr2[i]);
		}
	});

	ade::TensptrT gright = llo::derive(dest, src2.get());
	llo::GenericData gout_right = llo::eval(gright, age::INT32);
	ASSERT_EQ(age::INT32, gout_right.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout_right.shape_.begin(), gout_right.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	int32_t* goptr3 = (int32_t*) gout_right.data_.get();
	int_verify(sess, "gout_right", std::vector<int32_t>(goptr3, goptr3 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(bwd(data[i], data2[i], 0, 1), goptr3[i]);
		}
	});
}


TEST_F(API, Abs)
{
	simple::SessionT sess = get_session("API::Abs");
	unary_elementary(sess, default_range,
		[](ade::TensptrT& a) { return age::abs(a); },
		[](double d) { return std::abs(d); },
		[](double d) { return d / std::abs(d); }, false);
}


TEST_F(API, Neg)
{
	simple::SessionT sess = get_session("API::Neg");
	unary_elementary(sess, default_range,
		[](ade::TensptrT& a) { return age::neg(a); },
		[](double d) { return -d; },
		[](double d) { return -1.0; }, false);
}


TEST_F(API, Sin)
{
	simple::SessionT sess = get_session("API::Sin");
	unary_elementary(sess, default_range,
		[](ade::TensptrT& a) { return age::sin(a); },
		[](double d) { return std::sin(d); },
		[](double d) { return std::cos(d); });
}


TEST_F(API, Cos)
{
	simple::SessionT sess = get_session("API::Cos");
	unary_elementary(sess, default_range,
		[](ade::TensptrT& a) { return age::cos(a); },
		[](double d) { return std::cos(d); },
		[](double d) { return -std::sin(d); });
}


TEST_F(API, Tan)
{
	simple::SessionT sess = get_session("API::Tan");
	unary_elementary(sess, {-1, 1},
		[](ade::TensptrT& a) { return age::tan(a); },
		[](double d) { return std::tan(d); },
		[](double d) {
			double denom = std::cos(d);
			return 1.0 / denom / denom;
		});
}


TEST_F(API, Exp)
{
	simple::SessionT sess = get_session("API::Exp");
	unary_elementary(sess, {-9876, 5},
		[](ade::TensptrT& a) { return age::exp(a); },
		[](double d) { return std::exp(d); },
		[](double d) { return std::exp(d); });
}


TEST_F(API, Log)
{
	simple::SessionT sess = get_session("API::Log");
	unary_elementary(sess, {0.5, 9876},
		[](ade::TensptrT& a) { return age::log(a); },
		[](double d) { return std::log(d); },
		[](double d) { return 1.0 / d; });
}


TEST_F(API, Sqrt)
{
	simple::SessionT sess = get_session("API::Sqrt");
	unary_elementary(sess, {0, 9876},
		[](ade::TensptrT& a) { return age::sqrt(a); },
		[](double d) { return std::sqrt(d); },
		[](double d) { return 1.0 / (2 * std::sqrt(d)); });
}


TEST_F(API, Round)
{
	simple::SessionT sess = get_session("API::Round");
	unary_elementary(sess, default_range,
		[](ade::TensptrT& a) { return age::round(a); },
		[](double d) { return std::round(d); },
		[](double d) { return 1.0; }, false);
}


TEST_F(API, Flip)
{
	simple::SessionT sess = get_session("API::Flip");

	int32_t nrank = sess->get_scalar("nrank", {1, ade::rank_cap - 1});
	std::vector<ade::DimT> slist = get_shape_n(sess, nrank, "shape");
	ade::Shape shape(slist);
	uint8_t dim = 0;
	if (nrank > 1)
	{
		dim = sess->get_scalar("dim", {0, nrank - 1});
	}
	uint8_t baddim = sess->get_scalar("baddim", {nrank, ade::rank_cap});
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);

	ade::TensptrT src = llo::get_variable<double>(data, shape);
	ade::TensptrT dest = age::flip(src, dim);

	ade::TensptrT bad = age::flip(src, baddim);
	std::stringstream ss;
	ss << "attempting to flip dimension " <<
		(int) baddim << " beyond shape rank " << nrank;
	EXPECT_FATAL(llo::eval(bad, age::DOUBLE), ss.str().c_str())

	llo::GenericData out = llo::eval(dest, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, out.dtype_);
	std::vector<ade::DimT> gotshape(out.shape_.begin(), out.shape_.end());
	ASSERT_ARREQ(slist, gotshape);
	double* optr = (double*) out.data_.get();

	double_verify(sess, "out", std::vector<double>(optr, optr + n),
	[&]()
	{
		ade::CoordT coord;
		uint8_t dimlimit = shape.at(dim) - 1;
		for (size_t i = 0; i < n; ++i)
		{
			coord = ade::coordinate(shape, i);
			coord[dim] = dimlimit - coord[dim];

			EXPECT_EQ(data[ade::index(shape, coord)], optr[i]);
		}
	});

	ade::TensptrT gsrc = llo::derive(dest, src.get());

	llo::GenericData gout = llo::eval(gsrc, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST_F(API, Pow)
{
	simple::SessionT sess = get_session("API::Pow");
	binary_elementary(sess, {0.5, 5},
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::pow(a, b); },
		[](double a, double b) { return std::pow(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg * b * std::pow(a, b - 1) +
				rightg * std::pow(a, b) * std::log(a);
		});
}


TEST_F(API, Add)
{
	simple::SessionT sess = get_session("API::Add");
	binary_elementary(sess, default_range,
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::add(a, b); },
		[](double a, double b) { return a + b; },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg + rightg;
		});
}


TEST_F(API, Sub)
{
	simple::SessionT sess = get_session("API::Sub");
	binary_elementary(sess, default_range,
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::sub(a, b); },
		[](double a, double b) { return a - b; },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg - rightg;
		});
}


TEST_F(API, Mul)
{
	simple::SessionT sess = get_session("API::Mul");
	binary_elementary(sess, default_range,
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::mul(a, b); },
		[](double a, double b) { return a * b; },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg * b + rightg * a;
		});
}


TEST_F(API, Div)
{
	simple::SessionT sess = get_session("API::Div");
	binary_elementary(sess, default_range,
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::div(a, b); },
		[](double a, double b) { return a / b; },
		[](double a, double b, double leftg, double rightg)
		{
			return (leftg * b - rightg * a) / (b * b);
		});
}


TEST_F(API, Min)
{
	simple::SessionT sess = get_session("API::Min");
	binary_elementary(sess, default_range,
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::min({a, b}); },
		[](double a, double b) { return std::min(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			if (a > b)
			{
				return rightg;
			}
			else if (b > a)
			{
				return leftg;
			}
			// else
			return leftg + rightg;
		});
}


TEST_F(API, Max)
{
	simple::SessionT sess = get_session("API::Max");
	binary_elementary(sess, default_range,
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::max({a, b}); },
		[](double a, double b) { return std::max(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			if (a > b)
			{
				return leftg;
			}
			else if (b > a)
			{
				return rightg;
			}
			// else
			return leftg + rightg;
		});
}


TEST_F(API, Eq)
{
	simple::SessionT sess = get_session("API::Eq");
	binary_elementary_int(sess, {-1, 1},
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::eq(a, b); },
		[](int32_t a, int32_t b) { return a == b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST_F(API, Neq)
{
	simple::SessionT sess = get_session("API::Neq");
	binary_elementary_int(sess, {-1, 1},
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::neq(a, b); },
		[](int32_t a, int32_t b) { return a != b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST_F(API, Lt)
{
	simple::SessionT sess = get_session("API::Lt");
	binary_elementary_int(sess, {-1, 1},
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::lt(a, b); },
		[](int32_t a, int32_t b) { return a < b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST_F(API, Gt)
{
	simple::SessionT sess = get_session("API::Gt");
	binary_elementary_int(sess, {-1, 1},
		[](ade::TensptrT& a, ade::TensptrT& b) { return age::gt(a, b); },
		[](int32_t a, int32_t b) { return a > b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST_F(API, NElems)
{
	simple::SessionT sess = get_session("API::NElems");
	unary_generic(sess, default_range,
		[](ade::TensptrT& src) { return age::n_elems(src); },
		[&sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, out.shape_.n_elems());
			double got = *((double*) out.data_.get());

			double_verify(sess, "out", {got},
			[&]()
			{
				EXPECT_EQ(shape.n_elems(), got);
			});
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(0, gout[i]);
			}
		});
}


TEST_F(API, NDims)
{
	simple::SessionT sess = get_session("API::NDims");
	uint8_t dim = sess->get_scalar("dim", {0, ade::rank_cap - 1});

	unary_generic(sess, default_range,
		[dim](ade::TensptrT& src) { return age::n_dims(src, dim); },
		[dim, &sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, out.shape_.n_elems());
			double got = *((double*) out.data_.get());

			double_verify(sess, "out", {got},
			[&]()
			{
				EXPECT_EQ(shape.at(dim), got);
			});
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(0, gout[i]);
			}
		});
}


TEST_F(API, Rmax)
{
	simple::SessionT sess = get_session("API::Rmax");

	unary_generic(sess, default_range,
		[](ade::TensptrT& src) { return age::reduce_max(src); },
		[&sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = out.shape_.n_elems();
			ASSERT_EQ(1, n);
			double got = *((double*) out.data_.get());

			double_verify(sess, "out", {got},
			[&]()
			{
				double expect = *(std::max_element(data.begin(), data.end()));
				EXPECT_DOUBLE_EQ(expect, got);
			});
		},
		[](double* gout, std::vector<double>& og)
		{
			double bigly = *(std::max_element(og.begin(), og.end()));
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				if (og[i] == bigly)
				{
					EXPECT_EQ(1, gout[i]);
				}
			}
		});
}


TEST_F(API, Rsum)
{
	simple::SessionT sess = get_session("API::Rsum");

	unary_generic(sess, default_range,
		[](ade::TensptrT& src) { return age::reduce_sum(src); },
		[&sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = out.shape_.n_elems();
			{
				ASSERT_EQ(1, n);
			}
			double got = *((double*) out.data_.get());

			double_verify(sess, "out", {got},
			[&]()
			{
				double expect = std::accumulate(data.begin(), data.end(), 0.0);
				EXPECT_DOUBLE_EQ(expect, got);
			});
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(1, gout[i]);
			}
		});
}


TEST_F(API, Matmul)
{
	simple::SessionT sess = get_session("API::Matmul2d");

	ade::DimT cdim = sess->get_scalar("cdim", {1, 17});
	ade::DimT adim = sess->get_scalar("adim", {1, 17});
	ade::DimT bdim = sess->get_scalar("bdim", {1, 13});
	std::vector<ade::DimT> alist = {cdim, adim};
	std::vector<ade::DimT> blist = {bdim, cdim};
	std::vector<ade::DimT> sqrlist = {cdim, cdim};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape(sqrlist);

	ade::NElemT na = ashape.n_elems();
	ade::NElemT nb = bshape.n_elems();
	std::vector<int32_t> data = sess->get_int("data", na, {-9876, 9876});
	std::vector<int32_t> data2 = sess->get_int("data2", nb, {-9876, 9876});
	std::vector<int32_t> data3 = sess->get_int("data3", cdim * cdim, {-9876, 9876});

	ade::TensptrT a = llo::get_variable<int32_t>(data, ashape);
	ade::TensptrT b = llo::get_variable<int32_t>(data2, bshape);
	ade::TensptrT dest = age::matmul(a, b);

	llo::GenericData out = llo::eval(dest, age::INT32);
	EXPECT_EQ(age::INT32, out.dtype_);
	ade::Shape& gotshape = out.shape_;
	EXPECT_EQ(bdim, gotshape.at(0));
	EXPECT_EQ(adim, gotshape.at(1));
	int32_t* optr = (int32_t*) out.data_.get();
	ASSERT_NE(nullptr, optr);
	int_verify(sess, "out",
	std::vector<int32_t>(optr, optr + gotshape.n_elems()),
	[&]()
	{
		llo::GenericData ad = llo::eval(a, age::INT32);
		llo::GenericData bd = llo::eval(b, age::INT32);
		MatVecT dda = create_2d(ad);
		MatVecT ddb = create_2d(bd);
		MatVecT ddc = create_2d(out);
		EXPECT_TRUE(freivald(dda, ddb, ddc));
	});

	ade::TensptrT c = llo::get_variable<int32_t>(data3, cshape);
	ade::TensptrT dest2 = age::matmul(c, c);
	ade::TensptrT gsame = llo::derive(dest2, c.get());
	llo::GenericData gout = llo::eval(gsame, age::INT32);
	EXPECT_EQ(age::INT32, gout.dtype_);
	ade::Shape& gcshape = gout.shape_;
	{
		std::vector<ade::DimT> glist(gcshape.begin(), gcshape.end());
		ASSERT_ARREQ(sqrlist, glist);
	}
	int32_t* goptr = (int32_t*) gout.data_.get();

	// int_verify(sess, "gout",
	// std::vector<int32_t>(goptr, goptr + gcshape.n_elems()),
	// [&]()
	// {
	// 	// todo: implement
	// });

	ade::TensptrT gleft = llo::derive(dest, a.get());
	llo::GenericData gout_left = llo::eval(gleft, age::INT32);
	EXPECT_EQ(age::INT32, gout_left.dtype_);
	ade::Shape& gashape = gout_left.shape_;
	{
		std::vector<ade::DimT> glist(gashape.begin(), gashape.end());
		ASSERT_ARREQ(alist, glist);
	}
	int32_t* goptr2 = (int32_t*) gout_left.data_.get();

	// int_verify(sess, "gout_left",
	// std::vector<int32_t>(goptr2, goptr2 + gashape.n_elems()),
	// [&]()
	// {
	// 	// todo: implement
	// });

	ade::TensptrT gright = llo::derive(dest, b.get());
	llo::GenericData gout_right = llo::eval(gright, age::INT32);
	EXPECT_EQ(age::INT32, gout_right.dtype_);
	ade::Shape& gbshape = gout_right.shape_;
	{
		std::vector<ade::DimT> glist(gbshape.begin(), gbshape.end());
		ASSERT_ARREQ(blist, glist);
	}
	int32_t* goptr3 = (int32_t*) gout_right.data_.get();

	// int_verify(sess, "gout_right",
	// std::vector<int32_t>(goptr3, goptr3 + gbshape.n_elems()),
	// [&]()
	// {
	// 	// todo: implement
	// });
}


TEST_F(API, Permute)
{
	simple::SessionT sess = get_session("API::Permute");

	int32_t nrank = sess->get_scalar("nrank", {2, ade::rank_cap - 2});
	std::vector<ade::DimT> slist = get_shape_n(sess, nrank, "slist");
	std::vector<uint64_t> pidx_temp = sess->choose("pidx", slist.size(), slist.size());
	std::vector<uint8_t> pidx(pidx_temp.begin(), pidx_temp.end());
	ade::Shape shape(slist);
	ade::NElemT nelem = shape.n_elems();
	std::vector<double> data = sess->get_double("data", nelem, default_range);

	ade::TensptrT src = llo::get_variable<double>(data, shape);
	ade::TensptrT dest = age::permute(src, pidx);

	llo::GenericData out = llo::eval(dest, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, out.dtype_);
	size_t n = out.shape_.n_elems();
	ASSERT_EQ(nelem, n);
	double* got = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>{got, got + n},
	[&]()
	{
		ade::CoordT coord, temp;
		for (size_t i = 0; i < n; ++i)
		{
			coord = temp = ade::coordinate(shape, i);
			for (int32_t j = 0; j < nrank; ++j)
			{
				coord[j] = temp[pidx[j]];
			}

			EXPECT_EQ(data[i], got[ade::index(out.shape_, coord)]);
		}
	});

	ade::TensptrT gsrc = llo::derive(dest, src.get());

	llo::GenericData gout = llo::eval(gsrc, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST_F(API, Extend)
{
	simple::SessionT sess = get_session("API::Extend");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");

	int32_t nrank = slist.size();
	int32_t remainder = ade::rank_cap - nrank;

	int32_t n_ext = 1;
	if (remainder > 1)
	{
		n_ext = sess->get_scalar("n_ext", {1, remainder});
	}
	std::vector<ade::DimT> ext = get_shape_n(sess, n_ext, "ext");
	ade::Shape shape(slist);
	ade::NElemT nelem = shape.n_elems();
	std::vector<double> data = sess->get_double("data", nelem, default_range);

	ade::TensptrT src = llo::get_variable<double>(data, shape);
	ade::TensptrT dest = age::extend(src, nrank, ext);

	llo::GenericData out = llo::eval(dest, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, out.dtype_);
	size_t ext_nelem = ade::Shape(ext).n_elems();
	size_t n = out.shape_.n_elems();
	ASSERT_EQ(nelem * ext_nelem, n);
	double* got = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>{got, got + n},
	[&]()
	{
		for (size_t i = 0; i < nelem; ++i)
		{
			for (size_t j = 0; j < ext_nelem; ++j)
			{
				EXPECT_EQ(data[i], got[i + j * nelem]);
			}
		}
	});

	ade::TensptrT gsrc = llo::derive(dest, src.get());

	llo::GenericData gout = llo::eval(gsrc, age::DOUBLE);
	ASSERT_EQ(age::DOUBLE, gout.dtype_);
	{
		std::vector<ade::DimT> gotshape(gout.shape_.begin(), gout.shape_.end());
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0; i < nelem; ++i)
	{
		EXPECT_EQ(ext_nelem, goptr[i]);
	}
}


#endif // DISABLE_API_TEST
