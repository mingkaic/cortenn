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

/// Tree node for labeling Tensptrs
struct PathedTens final
{
	~PathedTens (void)
	{
		for (auto& child : children_)
		{
			delete child.second;
		}
	}

	/// Grab all leaf and branch nodes from subtree root other
	/// Accounting for duplicate labels but not Tensptrs
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

	/// Return tensor associated with input path if found otherwise nullptr
	ade::TensptrT get_labelled (StringsT path) const
	{
		return get_labelled(path.begin(), path.end());
	}

	/// Set input path to reference tensor
	void set_labelled (StringsT path, ade::TensptrT tens)
	{
		set_labelled(path.begin(), path.end(), tens);
	}

	/// Return tensor associated with path between iterators begin and end
	/// if found otherwise nullptr
	ade::TensptrT get_labelled (
		StringsT::iterator path_begin,
		StringsT::iterator path_end) const
	{
		if (path_begin == path_end)
		{
			return nullptr;
		}
		auto path_it = path_begin++;
		if (path_begin == path_end)
		{
			auto it = tens_.find(*path_it);
			if (tens_.end() != it)
			{
				return it->second;
			}
		}
		else
		{
			auto it = children_.find(*path_it);
			if (nullptr != it->second)
			{
				return it->second->get_labelled(path_begin, path_end);
			}
		}
		return nullptr;
	}

	/// Set path between iterators begin and end to reference tensor
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

	/// Map of labels to branching nodes
	std::unordered_map<std::string,PathedTens*> children_;

	/// Map of labels to tensor leaves
	std::unordered_map<std::string,ade::TensptrT> tens_;
};

/// Contains all information necessary to recreate labelled ADE graph
struct GraphInfo final
{
	/// Set of all roots (Tensptrs without any parent)
	std::unordered_set<ade::TensptrT> roots_;

	/// Labelled tensors
	PathedTens tens_;
};

/// Return graph info through out available from in graph
void load_graph (GraphInfo& out, const tenncor::Graph& in,
	DataLoaderT dataloader);

}

#endif // PBM_GRAPH_HPP
