#include "ade/ileaf.hpp"

#ifndef MOCK_CODES_DEP_HPP
#define MOCK_CODES_DEP_HPP

struct Meat
{
	Meat (size_t idx) : num(idx), size(idx / 2) {}

	int64_t num;
	uint64_t size;
};

struct Fries
{
	Fries (size_t idx) : num(idx / 3), size(idx * 2) {}

	double num;
	char size;
};

ade::TensptrT cooler (size_t bardock);

ade::TensptrT freeza (ade::TensptrT a, uint8_t bardock);

#endif // MOCK_CODES_DEP_HPP
