#include "ade/ileaf.hpp"

#include "llo/serialize.hpp"

#ifdef LLO_SERIALIZE_HPP

namespace llo
{

bool is_big_endian(void)
{
	union
	{
		uint16_t _;
		char bytes[2];
	} twob = { 0x0001 };

	return twob.bytes[0] == 0;
}

std::string serialize (const char* in, size_t nelems, size_t typecode)
{
	size_t nbytes = age::type_size((age::_GENERATED_DTYPE) typecode);
	if (is_big_endian() && nbytes > 1)
	{
		size_t totalbytes = nelems * nbytes;
		std::string out(totalbytes, '\0');
		for (size_t i = 0; i < totalbytes; ++i)
		{
			size_t elemi = i / nbytes;
			size_t outi = (elemi + 1) * nbytes - (i % nbytes);
			out[outi] = in[i];
		}
		return out;
	}
	return std::string(in, nelems * nbytes);
}

inline ade::TensptrT variable_from_code (const char* cptr,
	age::_GENERATED_DTYPE dtype, ade::Shape shape, std::string label)
{
	switch (dtype)
	{
		case age::DOUBLE:
			return ade::TensptrT(new Variable<double>(
				(double*) cptr, shape, label));
		case age::FLOAT:
			return ade::TensptrT(new Variable<float>(
				(float*) cptr, shape, label));
		case age::INT8:
			return ade::TensptrT(new Variable<int8_t>(
				(int8_t*) cptr, shape, label));
		case age::INT16:
			return ade::TensptrT(new Variable<int16_t>(
				(int16_t*) cptr, shape, label));
		case age::INT32:
			return ade::TensptrT(new Variable<int32_t>(
				(int32_t*) cptr, shape, label));
		case age::INT64:
			return ade::TensptrT(new Variable<int64_t>(
				(int64_t*) cptr, shape, label));
		case age::UINT8:
			return ade::TensptrT(new Variable<uint8_t>(
				(uint8_t*) cptr, shape, label));
		case age::UINT16:
			return ade::TensptrT(new Variable<uint16_t>(
				(uint16_t*) cptr, shape, label));
		case age::UINT32:
			return ade::TensptrT(new Variable<uint32_t>(
				(uint32_t*) cptr, shape, label));
		case age::UINT64:
			return ade::TensptrT(new Variable<uint64_t>(
				(uint64_t*) cptr, shape, label));
		default:
			logs::fatalf("unknown dtype \"%s\"",
				age::name_type(dtype).c_str());
	}
}

ade::TensptrT deserialize (const char* pb, ade::Shape shape,
	size_t typecode, std::string label)
{
	age::_GENERATED_DTYPE gencode = (age::_GENERATED_DTYPE) typecode;
	size_t nbytes = age::type_size(gencode);
	if (is_big_endian() && nbytes > 1)
	{
		size_t totalbytes = shape.n_elems() * nbytes;
		std::string out(totalbytes, '\0');
		for (size_t i = 0; i < totalbytes; ++i)
		{
			size_t elemi = i / nbytes;
			size_t outi = (elemi + 1) * nbytes - (i % nbytes);
			out[outi] = pb[i];
		}
		return variable_from_code(out.c_str(), gencode, shape, label);
	}
	return variable_from_code(pb, gencode, shape, label);
}

}

#endif
