#include "llo/operator.hpp"

#include "llo/opt/ops_merge.hpp"

#ifdef LLO_OPS_MERGE_HPP

namespace llo
{

static bool is_bijective (ade::CoordptrT coorder)
{
	return ade::identity == coorder ||
		coorder->is_bijective();
}

ade::TensptrT ops_merge_edit (bool& is_optimized,
	ade::Opcode& opcode, ade::ArgsT& args)
{
	if (nnary_codes.end() != nnary_codes.find(
		(age::_GENERATED_OPCODE) opcode.code_))
	{
		bool merged = false;
		ade::ArgsT newchildren;
		for (ade::FuncArg& arg : args)
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
			bool arg_id = is_bijective(arg_coorder);
			if (opcode.code_ == arg_op && (arg_id || std::all_of(
				arg_children.begin(), arg_children.end(),
				[](ade::FuncArg& mten)
				{
					return is_bijective(mten.get_coorder());
				})))
			{
				// merge child's mapper and shaper to these arguments
				for (auto child : arg_children)
				{
					bool newdirection;
					auto acoorder = arg_coorder;
					bool child_io = child.map_io();
					auto ccoorder = child.get_coorder();
					// arg is identity, so take children's direction and coorder
					if (arg_id)
					{
						newdirection = child_io;
						acoorder = newdirection != arg_io ?
							ade::CoordptrT(acoorder->reverse()) :
							acoorder;
					}
					// child.get_coorder() is identity,
					// so take arg's direction and coorder
					else
					{
						newdirection = arg_io;
						ccoorder = newdirection != child_io ?
							ade::CoordptrT(ccoorder->reverse()) :
							ccoorder;
					}
					newchildren.push_back(ade::FuncArg(
						child.get_tensor(),
						ade::CoordptrT(child.get_shaper()->
							connect(*arg_shaper)),
						newdirection,
						ade::CoordptrT(newdirection ?
							ccoorder->connect(*acoorder) : // child -> arg
							acoorder->connect(*ccoorder)) // arg -> child
					));
				}
				merged = true;
			}
			else if (arg_children.size() == 1 &&
				nnary_codes.end() != nnary_codes.find(arg_op) &&
				arg_children[0].get_coorder()->is_bijective())
			{
				auto child = arg_children[0];
				bool child_io = child.map_io();
				auto ccoorder = child.get_coorder();
				ccoorder = arg_io != child_io ?
					ade::CoordptrT(ccoorder->reverse()) :
					ccoorder;
				newchildren.push_back(ade::FuncArg(
					child.get_tensor(),
					ade::CoordptrT(child.get_shaper()->
						connect(*arg_shaper)),
					arg_io,
					ade::CoordptrT(arg_io ?
						ccoorder->connect(*arg_coorder) : // child -> arg
						arg_coorder->connect(*ccoorder)) // arg -> child
				));
				merged = true;
			}
			else
			{
				newchildren.push_back(arg);
			}
		}
		if (1 == newchildren.size() &&
			ade::is_identity(newchildren[0].get_coorder().get()))
		{
			return newchildren[0].get_tensor();
		}
		else if (merged)
		{
			is_optimized = true;
			args = newchildren;
		}
	}
	return nullptr;
}

ade::TensT ops_merge (ade::TensT roots)
{
	return opt::graph_edit(roots,
		[](ade::Opcode& opcode,
			ade::ArgsT& args, bool changed) -> ade::TensptrT
		{
			bool is_optimized = false;
			if (auto out = ops_merge_edit(is_optimized, opcode, args))
			{
				return out;
			}
			if (changed || is_optimized)
			{
				return ade::TensptrT(ade::Functor::get(opcode, args));
			}
			return nullptr;
		});
}

}

#endif
