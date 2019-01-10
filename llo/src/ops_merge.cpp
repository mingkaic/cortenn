#include "llo/opt/ops_merge.hpp"

#ifdef LLO_OPMERGE_HPP

namespace llo
{

static const std::unordered_set<age::_GENERATED_OPCODE> nnary = {
	age::SUM,
	age::PROD,
	age::MIN,
	age::MAX,
};

static bool is_identity (ade::CoordptrT coorder)
{
	if (ade::identity == coorder)
	{
		return true;
	}
	bool eq = true;
	coorder->access([&](const ade::MatrixT& m)
	{
		for (uint8_t i = 0; eq && i < ade::mat_dim; ++i)
		{
			for (uint8_t j = 0; eq && j < i; ++j)
			{
				eq = 0 == m[i][j];
			}
			eq = 1 == m[i][i];
			for (uint8_t j = i + 1; eq && j < ade::mat_dim; ++j)
			{
				eq = 0 == m[i][j];
			}
		}
	});
	return eq;
}

static ade::TensptrT ops_merge_edit (ade::iFunctor* func, ade::ArgsT args)
{
	ade::Opcode fopcode = func->get_opcode();
	age::_GENERATED_OPCODE opcode = (age::_GENERATED_OPCODE) fopcode.code_;
	if (nnary.end() != nnary.find(opcode))
	{
		bool merged = false;
		ade::ArgsT newchildren;
		for (ade::MappedTensor& arg : args)
		{
			ade::iTensor* argptr = arg.get_tensor().get();
			auto arg_shaper = arg.get_shaper();
			bool arg_io = arg.map_io();
			auto arg_coorder = arg.get_coorder();

			auto f = dynamic_cast<ade::iFunctor*>(argptr);
			ade::ArgsT arg_children;
			age::_GENERATED_OPCODE arg_op = age::BAD_OP;
			if (nullptr != f)
			{
				arg_children = f->get_children();
				arg_op = (age::_GENERATED_OPCODE) f->get_opcode().code_;
			}
			bool arg_id = is_identity(arg_coorder);
			// merge with arg_children if:
			// - arg or its children use identity coorders
			// - arg has same opcode as func OR arg is nnary with only a child
			if ((opcode == arg_op && (arg_id || std::all_of(
				arg_children.begin(), arg_children.end(),
				[](ade::MappedTensor& mten)
				{
					return is_identity(mten.get_coorder());
				}))) ||
				(arg_children.size() == 1 &&
				nnary.end() != nnary.find(arg_op) &&
				is_identity(arg_children[0].get_coorder())))
			{
				bool newdirection;
				ade::CoordptrT newcoorder;
				// merge child's mapper and shaper to these arguments
				for (auto child : arg_children)
				{
					// arg is identity, so take children's direction and coorder
					if (arg_id)
					{
						newdirection = child.map_io();
						newcoorder = child.get_coorder();
					}
					// child.get_coorder() is identity,
					// so take arg's direction and coorder
					else
					{
						newdirection = arg_io;
						newcoorder = arg_coorder;
					}
					newchildren.push_back(ade::MappedTensor(
						child.get_tensor(),
						ade::CoordptrT(child.get_shaper()->
							connect(*arg_shaper)),
						newdirection,
						newcoorder
					));
				}
				merged = true;
			}
			else
			{
				newchildren.push_back(arg);
			}
		}
		if (merged)
		{
			return ade::TensptrT(ade::Functor::get(fopcode, newchildren));
		}
	}
	return nullptr;
}

ade::TensptrT ops_merge (ade::TensptrT root)
{
	opt::GraphEditor opedit(ops_merge_edit);
	return opedit.edit(root);
}

}

#endif
