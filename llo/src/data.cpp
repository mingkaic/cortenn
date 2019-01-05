#include "llo/data.hpp"

#ifdef LLO_DATA_HPP

namespace llo
{

struct CDeleter
{
	void operator () (void* p)
	{
		free(p);
	}
};

GenericData::GenericData (ade::Shape shape, age::_GENERATED_DTYPE dtype) :
	data_((char*) malloc(shape.n_elems() * type_size(dtype)),
		CDeleter()), shape_(shape), dtype_(dtype) {}

#define COPYOVER(TYPE) { std::vector<TYPE> temp(indata, indata + n);\
	std::memcpy(out, &temp[0], nbytes); } break;

template <typename T>
void convert (char* out, age::_GENERATED_DTYPE outtype, const T* indata, size_t n)
{
	size_t nbytes = type_size(outtype) * n;
	switch (outtype)
	{
		case age::DOUBLE: COPYOVER(double)
		case age::FLOAT: COPYOVER(float)
		case age::INT8: COPYOVER(int8_t)
		case age::INT16: COPYOVER(int16_t)
		case age::INT32: COPYOVER(int32_t)
		case age::INT64: COPYOVER(int64_t)
		case age::UINT8: COPYOVER(uint8_t)
		case age::UINT16: COPYOVER(uint16_t)
		case age::UINT32: COPYOVER(uint32_t)
		case age::UINT64: COPYOVER(uint64_t)
		default: logs::fatalf("invalid output type %s",
			age::name_type(outtype).c_str());
	}
}

#undef COPYOVER

#define CONVERT(INTYPE)\
convert<INTYPE>(data_.get(), dtype_, (const INTYPE*) indata, n); break;

void GenericData::copyover (const char* indata, age::_GENERATED_DTYPE intype)
{
	size_t n = shape_.n_elems();
	if (dtype_ == intype)
	{
		std::memcpy(data_.get(), indata, type_size(dtype_) * n);
	}
	switch (intype)
	{
		case age::DOUBLE: CONVERT(double)
		case age::FLOAT: CONVERT(float)
		case age::INT8: CONVERT(int8_t)
		case age::INT16: CONVERT(int16_t)
		case age::INT32: CONVERT(int32_t)
		case age::INT64: CONVERT(int64_t)
		case age::UINT8: CONVERT(uint8_t)
		case age::UINT16: CONVERT(uint16_t)
		case age::UINT32: CONVERT(uint32_t)
		case age::UINT64: CONVERT(uint64_t)
		default: logs::fatalf("invalid input type %s",
			age::name_type(intype).c_str());
	}
}

#undef CONVERT

}

#endif
