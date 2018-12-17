///
/// graph.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal equation graph
///

#include "pbm/data.hpp"

#ifndef PBM_LOAD_HPP
#define PBM_LOAD_HPP

namespace pbm
{

struct PathedTens final
{
	~PathedTens (void)
	{
		for (auto& child : children_)
		{
			delete child.second;
		}
	}

	void join (PathedTens* other)
	{
		{
			std::vector<std::string> labels;
			for (auto opair : other->tens_)
			{
				if (tens_.end() != tens_.find(opair.first))
				{
					labels.push_back(opair.first);
				}
			}
			if (tens_.size() > 0)
			{
				logs::warnf("duplicate base labels %s",
					fmts::to_string(labels.begin(), labels.end()).c_str());
			}
		}

		for (auto cpair : other->children_)
		{
			std::string label = cpair.first;
			auto it = children_.find(label);
			if (children_.end() == it)
			{
				children_.emplace(label, cpair.second);
			}
			else
			{
				it->second->join(cpair.second);
			}
		}
	}

	ade::TensptrT get_labelled (
		StringsT::iterator path_begin,
		StringsT::iterator path_end) const
	{
		if (path_begin == path_end)
		{
			return nullptr;
		}
		auto it = children_.find(*path_begin);
		assert(nullptr != it->second);
		return it->second->get_labelled(++path_begin, path_end);
	}

	void set_labelled (StringsT::iterator path_begin,
		StringsT::iterator path_end, ade::TensptrT tens)
	{
		if (path_begin == path_end)
		{
			return;
		}
		std::string label = *(path_begin++);
		if (path_begin == path_end)
		{
			tens_.emplace(label, tens);
			return;
		}
		PathedTens* child;
		auto it = children_.find(label);
		if (it == children_.end())
		{
			child = new PathedTens();
			children_.emplace(label, child);
		}
		else
		{
			assert(nullptr != it->second);
			child = it->second;
		}
		child->set_labelled(path_begin, path_end, tens);
	}

	std::unordered_map<std::string,PathedTens*> children_;

	std::unordered_map<std::string,ade::TensptrT> tens_;
};

struct GraphInfo final
{
	std::unordered_set<ade::TensptrT> roots_;

	PathedTens tens_;
};

/// Return all nodes in graph unmarshalled from protobuf object
void load_graph (GraphInfo& out, const tenncor::Graph& in,
	DataLoaderT dataloader);

}

#endif // PBM_GRAPH_HPP
