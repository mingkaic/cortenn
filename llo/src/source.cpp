#include "ade/ileaf.hpp"

#include "llo/source.hpp"

#ifdef LLO_SOURCE_HPP

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
        return ade::TensptrT(new Variable(out.c_str(), gencode, shape, label));
    }
    return ade::TensptrT(new Variable(pb, gencode, shape, label));
}

}

#endif
