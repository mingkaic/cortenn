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

LoadVecsT load_graph (const tenncor::Graph& in, iDataLoader& dataloader)
{
	auto nodes = in.nodes();
	TensT invec;
	LoadVecsT outvec;
	for (const tenncor::Node& node : nodes)
	{
		auto pb_labels = node.labels();
		if (node.has_source())
		{
			std::string src_label = *(pb_labels.rbegin());
			const tenncor::Source& source = node.source();
			ade::TensptrT leaf = dataloader.load(source, src_label);
			invec.push_back(leaf);
			if (false == pb_labels.empty())
			{
				StringsT labels(pb_labels.begin(), pb_labels.end());
				outvec.push_back({leaf, labels});
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
				args.push_back({coord, invec[nodearg.idx()]});
			}
			ade::TensptrT f(ade::Functor::get(
				ade::Opcode{func.opname(), func.opcode()}, args));
			invec.push_back(f);
			if (false == pb_labels.empty())
			{
				StringsT labels(pb_labels.begin(), pb_labels.end());
				outvec.push_back({f, labels});
			}
		}
	}
	return outvec;
}

}

#endif
