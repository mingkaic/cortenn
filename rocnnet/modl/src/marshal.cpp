#include "rocnnet/modl/marshal.hpp"

#ifdef MODL_MARSHAL_HPP

namespace modl
{

void save (std::ostream& outs, ade::TensptrT& source,
	iMarshaler* source_graph)
{
	pbm::GraphSaver saver(llo::serialize);
	source->accept(saver);

	tenncor::Graph graph;
	saver.save(graph, source_graph->list_bases());
	graph.SerializeToOstream(&outs);
}

void load (std::istream& ins, iMarshaler* target)
{
	tenncor::Graph graph;
	graph.ParseFromIstream(&ins);
	pbm::GraphInfo info;
	pbm::load_graph(info, graph, llo::deserialize);
	target->set_variables(&info.tens_);
}

}

#endif
