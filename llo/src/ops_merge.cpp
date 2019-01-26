#include <set>

// #include "coord/coord.hpp"

#include "llo/opt/ops_merge.hpp"

#ifdef LLO_OPMERGE_HPP

namespace llo
{

// todo: make bool array once generator expresses number of OPCODEs
static const std::set<age::_GENERATED_OPCODE> nnary = {
	age::SUM,
	age::PROD,
	age::MIN,
	age::MAX,
};

static bool is_bijective (ade::CoordptrT coorder)
{
	return ade::identity == coorder ||
		coorder->is_bijective();
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
			bool arg_id = is_bijective(arg_coorder);
			if (opcode == arg_op && (arg_id || std::all_of(
				arg_children.begin(), arg_children.end(),
				[](ade::MappedTensor& mten)
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
					newchildren.push_back(ade::MappedTensor(
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
				nnary.end() != nnary.find(arg_op) &&
				is_bijective(arg_children[0].get_coorder()))
			{
				auto child = arg_children[0];
				bool child_io = child.map_io();
				auto ccoorder = child.get_coorder();
				ccoorder = arg_io != child_io ?
					ade::CoordptrT(ccoorder->reverse()) :
					ccoorder;
				newchildren.push_back(ade::MappedTensor(
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
