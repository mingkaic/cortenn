///
/// source.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "ade/ileaf.hpp"

#include "pbm/data.hpp"

#include "llo/data.hpp"

#ifndef LLO_SOURCE_HPP
#define LLO_SOURCE_HPP

namespace llo
{

constexpr bool is_big_endian(void)
{
    union
    {
        uint16_t _;
        char bytes[2];
    } twob = { 0x0001 };

    return twob.bytes[0] == 0;
}

/// Marshal iSource to tenncor::Source
struct DataSaver final : public pbm::iDataSaver
{
    std::string serialize (const char* in,
        size_t nelems, size_t typecode) override
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
};

#define UNPACK_SOURCE(TYPE)\
auto vec = arr.data();\
return ade::TensptrT(get_variable(std::vector<TYPE>(vec.begin(), vec.end()), shape, label));

/// Unmarshal tenncor::Source as Variable containing context of source
struct DataLoader final : public pbm::iDataLoader
{
    ade::TensptrT deserialize (const char* pb,
        ade::Shape shape, size_t typecode, std::string label) override
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
};

#undef UNPACK_SOURCE

}

#endif // LLO_SOURCE_HPP
