#include "err/log.hpp"

#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#include "pbm/load.hpp"

#ifdef PBM_LOAD_HPP

namespace pbm
{

static ade::CoordPtrT load_coord (
	const google::protobuf::RepeatedField<double>& coord)
{
	if (ade::mat_dim * ade::mat_dim != coord.size())
	{
		err::fatal("cannot deserialize non-matrix coordinate map");
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

LabelTensT load_graph (const tenncor::Graph& in, DataLoaderPtrT dataloader)
{
	auto nodes = in.nodes();
	TensT outvec;
	LabelTensT outmap;
	for (const tenncor::Node& node : nodes)
	{
		std::string label = node.label();
		if (node.has_source())
		{
			const tenncor::Source& source = node.source();
			ade::Tensorptr leaf = dataloader->load(source);
			outvec.push_back(leaf);
			if (false == label.empty())
			{
				outmap.emplace(label, leaf);
			}
		}
		else
		{
			tenncor::Functor func = node.functor();
			auto nodeargs = func.args();
			ade::ArgsT args;
			for (auto nodearg : nodeargs)
			{
				ade::CoordPtrT coord = load_coord(nodearg.coord());
				args.push_back({coord, outvec[nodearg.idx()]});
			}
			ade::Tensorptr f = ade::Functor::get(
				ade::Opcode{func.opname(), func.opcode()}, args);
			outvec.push_back(f);
			if (false == label.empty())
			{
				outmap.emplace(label, f);
			}
		}
	}
	return outmap;
}

}

#endif
