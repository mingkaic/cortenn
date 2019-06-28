#include "gtest/gtest.h"

#include "logs/logs.hpp"

#include "fmts/fmts.hpp"

#include "dbg/stream/ade.hpp"

#ifndef TESTUTIL_COMMON_HPP
#define TESTUTIL_COMMON_HPP

std::string compare_graph (std::istream& expectstr, ade::TensptrT root,
	bool showshape = true, LabelsMapT labels = {});

#define EXPECT_GRAPHEQ(MSG, ROOT) {\
	std::istringstream ss(MSG);\
	auto compare_str = compare_graph(ss, ROOT);\
	EXPECT_EQ(0, compare_str.size()) << compare_str;\
}

#endif // TESTUTIL_COMMON_HPP
