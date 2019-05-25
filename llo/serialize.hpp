///
/// serialize.hpp
/// llo
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "ade/functor.hpp"

#include "pbm/data.hpp"

#include "llo/generated/opmap.hpp"

#include "llo/variable.hpp"
#include "llo/constant.hpp"

#ifndef LLO_SERIALIZE_HPP
#define LLO_SERIALIZE_HPP

namespace llo
{

static bool is_big_endian(void)
{
	union
	{
		uint16_t _;
		char bytes[2];
	} twob = { 0x0001 };

	return twob.bytes[0] == 0;
}

struct LLOSaver : public pbm::iSaver
{
	std::string save_leaf (bool& is_const, ade::iLeaf* leaf) override
	{
		is_const = nullptr != dynamic_cast<Constant*>(leaf);

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

	std::vector<double> save_shaper (const ade::CoordptrT& mapper) override
	{
		std::vector<double> out;
		mapper->access(
			[&out](const ade::MatrixT& mat)
			{
				for (uint8_t i = 0; i < ade::mat_dim; ++i)
				{
					for (uint8_t j = 0; j < ade::mat_dim; ++j)
					{
						out.push_back(mat[i][j]);
					}
				}
			});
		return out;
	}

	std::vector<double> save_coorder (const ade::CoordptrT& mapper) override
	{
		return save_shaper(mapper);
	}
};

/// Unmarshal cortenn::Source as Variable containing context of source
struct LLOLoader : public pbm::iLoader
{

#define _SET_VAL(realtype)out_tens = ade::TensptrT(\
Variable<realtype>::get((realtype*) pb, shape, label));

	ade::TensptrT generate_leaf (const char* pb, ade::Shape shape,
		std::string typelabel, std::string label, bool is_const) override
	{
		ade::TensptrT out_tens;
		age::_GENERATED_DTYPE gencode = (age::_GENERATED_DTYPE) age::get_type(typelabel);
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
			if (is_const)
			{
				out_tens = ade::TensptrT(Constant::get(out.c_str(), gencode, shape));
			}
			else
			{
				pb = out.c_str();
				TYPE_LOOKUP(_SET_VAL, gencode)
			}
		}
		else if (is_const)
		{
			out_tens = ade::TensptrT(Constant::get(pb, gencode, shape));
		}
		else
		{
			TYPE_LOOKUP(_SET_VAL, gencode)
		}
		return out_tens;
	}

#undef _SET_VAL

	ade::TensptrT generate_func (std::string opname, ade::ArgsT args) override
	{
		return ade::TensptrT(ade::Functor::get(ade::Opcode{opname, age::get_op(opname)}, args));
	}

	ade::CoordptrT generate_shaper (std::vector<double> coord) override
	{
		if (ade::mat_dim * ade::mat_dim != coord.size())
		{
			logs::fatal("cannot deserialize non-matrix coordinate map");
		}
		return std::make_shared<ade::CoordMap>(
			[&](ade::MatrixT fwd)
			{
				for (uint8_t i = 0; i < ade::mat_dim; ++i)
				{
					for (uint8_t j = 0; j < ade::mat_dim; ++j)
					{
						fwd[i][j] = coord[i * ade::mat_dim + j];
					}
				}
			});
	}

	ade::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) override
	{
		return generate_shaper(coord);
	}
};

}

#endif // LLO_SERIALIZE_HPP
