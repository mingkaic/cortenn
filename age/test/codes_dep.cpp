#include "age/test/grader_dep.hpp"
#include "age/test/codes_dep.hpp"

ade::TensptrT cooler (size_t bardock)
{
	return ade::TensptrT(new MockTensor(bardock, ade::Shape(
		std::vector<ade::DimT>{(ade::DimT) bardock})));
}

ade::TensptrT freeza (ade::TensptrT a, uint8_t bardock)
{
	return ade::TensptrT(new MockTensor(bardock, ade::Shape(
		std::vector<ade::DimT>{(ade::DimT) a->shape().at(bardock)})));
}
