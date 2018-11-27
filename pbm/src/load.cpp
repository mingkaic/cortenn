#include "logs/logs.hpp"

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

void load_graph (GraphInfo& out, const tenncor::Graph& in,
	DataLoaderT dataloader)
{
	auto nodes = in.nodes();
	TensT invec;
	for (const tenncor::Node& node : nodes)
	{
		auto pb_labels = node.labels();
		if (node.has_source())
		{
			std::string src_label;
			if (pb_labels.size() > 0)
			{
				src_label = *(pb_labels.rbegin());
			}
			const tenncor::Source& source = node.source();
			std::string sstr = source.shape();
			ade::Shape shape(std::vector<ade::DimT>(sstr.begin(), sstr.end()));
			size_t tcode = source.typecode();
			std::string data = source.data();
			ade::TensptrT leaf = dataloader(data.c_str(),
				shape, tcode, src_label);
			invec.push_back(leaf);
			if (false == pb_labels.empty())
			{
				StringsT labels(pb_labels.begin(), pb_labels.end());
				out.labelled_.push_back({leaf, labels});
			}
			out.roots_.emplace(leaf);
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
				out.roots_.erase(invec[nodearg.idx()]);
			}
			ade::TensptrT f(ade::Functor::get(
				ade::Opcode{func.opname(), func.opcode()}, args));
			invec.push_back(f);
			if (false == pb_labels.empty())
			{
				StringsT labels(pb_labels.begin(), pb_labels.end());
				out.labelled_.push_back({f, labels});
			}
			out.roots_.emplace(f);
		}
	}
}

}

#endif
