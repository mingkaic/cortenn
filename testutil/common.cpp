#include "testutil/common.hpp"

std::string TestLogger::latest_warning_;
std::string TestLogger::latest_error_;

std::shared_ptr<TestLogger> tlogger = std::make_shared<TestLogger>();

std::vector<ade::DimT> get_shape_n (simple::SessionT& sess, size_t n, std::string label)
{
	int32_t max_elem = std::log(nelem_limit) / std::log(n);
	max_elem = std::max(3, max_elem);
	auto temp = sess->get_int(label, n, {2, max_elem});
	return std::vector<ade::DimT>(temp.begin(), temp.end());
}

std::vector<ade::DimT> get_shape (simple::SessionT& sess, std::string label)
{
	int32_t n = sess->get_scalar("n_" + label, {1, ade::rank_cap - 1});
	return get_shape_n(sess, n, label);
}

std::vector<ade::DimT> get_longshape (simple::SessionT& sess, std::string label)
{
	int32_t nl = sess->get_scalar("n_" + label, {ade::rank_cap, 57});
	return get_shape_n(sess, nl, label);
}

std::vector<ade::DimT> get_zeroshape (simple::SessionT& sess, std::string label)
{
	int32_t nz = sess->get_scalar("n_" + label, {1, ade::rank_cap - 1});
	int32_t max_zelem = std::log(nelem_limit) / std::log(nz);
	max_zelem = std::max(3, max_zelem);
	auto temp = sess->get_int(label, nz, {0, max_zelem});
	int32_t zidx = 0;
	if (nz > 1)
	{
		zidx = sess->get_scalar(label + "_idx", {0, nz - 1});
	}
	temp[zidx] = 0;
	return std::vector<ade::DimT>(temp.begin(), temp.end());
}

std::vector<ade::DimT> get_incompatible (simple::SessionT& sess,
	std::vector<ade::DimT> inshape, std::string label)
{
	int32_t rank = inshape.size();
	int32_t incr_pt = 0;
	if (rank > 1)
	{
		incr_pt = sess->get_scalar(label + "_incr_pt", {0, rank - 1});
	}
	std::vector<ade::DimT> bad = inshape;
	bad[incr_pt]++;
	return bad;
}

void int_verify (simple::SessionT& sess, std::string key,
	std::vector<int32_t> data, std::function<void()> verify)
{
	// if (sess->generated_input())
	// {
		verify();
		sess->store_int(key, data);
	// }
	// else
	// {
	// 	auto expect = expect_int(key);
	// 	EXPECT_ARREQ(expect, data);
	// }
}

void double_verify (simple::SessionT& sess, std::string key,
	std::vector<double> data, std::function<void()> verify)
{
	// if (sess->generated_input())
	// {
		verify();
		sess->store_double(key, data);
	// }
	// else
	// {
	// 	auto expect = expect_double(key);
	// 	EXPECT_ARREQ(expect, data);
	// }
}
