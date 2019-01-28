#include "ade/ileaf.hpp"

#include "llo/constant.hpp"
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

std::string serialize (bool& is_const, ade::iLeaf* leaf)
{
	is_const = nullptr != dynamic_cast<llo::Constant*>(leaf);

	char* data = (char*) leaf->data();
	size_t nelems = leaf->shape().n_elems();
	size_t nbytes = age::type_size((age::_GENERATED_DTYPE) leaf->type_code());
	if (is_big_endian() && nbytes > 1)
	{
		size_t totalbytes = nelems * nbytes;
		std::string out(totalbytes, '\0');
		for (size_t i = 0; i < totalbytes; ++i)
		{
			size_t elemi = i / nbytes;
			size_t outi = (elemi + 1) * nbytes - (i % nbytes);
			out[outi] = data[i];
		}
		return out;
	}
	return std::string(data, nelems * nbytes);
}

inline ade::iLeaf* variable_from_code (const char* cptr,
	age::_GENERATED_DTYPE dtype, ade::Shape shape, std::string label)
{
	switch (dtype)
	{
		case age::DOUBLE:
			return Variable<double>::get(
				(double*) cptr, shape, label);
		case age::FLOAT:
			return Variable<float>::get(
				(float*) cptr, shape, label);
		case age::INT8:
			return Variable<int8_t>::get(
				(int8_t*) cptr, shape, label);
		case age::INT16:
			return Variable<int16_t>::get(
				(int16_t*) cptr, shape, label);
		case age::INT32:
			return Variable<int32_t>::get(
				(int32_t*) cptr, shape, label);
		case age::INT64:
			return Variable<int64_t>::get(
				(int64_t*) cptr, shape, label);
		case age::UINT8:
			return Variable<uint8_t>::get(
				(uint8_t*) cptr, shape, label);
		case age::UINT16:
			return Variable<uint16_t>::get(
				(uint16_t*) cptr, shape, label);
		case age::UINT32:
			return Variable<uint32_t>::get(
				(uint32_t*) cptr, shape, label);
		case age::UINT64:
			return Variable<uint64_t>::get(
				(uint64_t*) cptr, shape, label);
		default:
			logs::fatalf("unknown dtype \"%s\"",
				age::name_type(dtype).c_str());
	}
}

ade::TensptrT deserialize (const char* pb, ade::Shape shape,
	size_t typecode, std::string label, bool is_const)
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
		return ade::TensptrT(is_const ?
			llo::Constant::get(out.c_str(), gencode, shape) :
			variable_from_code(out.c_str(), gencode, shape, label));
	}
	return ade::TensptrT(is_const ?
		llo::Constant::get(pb, gencode, shape) :
		variable_from_code(pb, gencode, shape, label));
}

}

#endif
