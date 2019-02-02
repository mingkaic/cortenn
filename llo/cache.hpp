#include "ade/ade.hpp"

#include "llo/variable.hpp"

#ifndef LLO_CACHE_HPP
#define LLO_CACHE_HPP

namespace llo
{

// todo: make path intersection finder more efficient
template <typename T>
struct CacheSpace final
{
	CacheSpace (ade::TensT roots)
	{
		add_caches(roots);
	}

	void set (ade::iFunctor* key, TensptrT<T> value)
	{
		if (caches_.end() != caches_.find(key))
		{
			caches_[key] = value;
			need_update_.erase(key);
		}
	}

	/// Return nullptr if key value needs updating
	TensptrT<T> get (ade::iFunctor* key) const
	{
		auto it = caches_.find(key);
		if (caches_.end() == it ||
			need_update_.end() != need_update_.find(key))
		{
			return nullptr;
		}
		return it->second;
	}

	/// Return true if cache contains value to functor key
	///	or the key needs updating
	bool has_value (ade::iFunctor* key) const
	{
		return caches_.end() != caches_.find(key);
	}

	void add_equation (ade::TensT roots)
	{
		add_caches(roots);
	}

	void mark_update (std::vector<llo::iVariable*> updates)
	{
		for (llo::iVariable* var : updates)
		{
			auto it = ancestors_.find(var);
			if (ancestors_.end() != it)
			{
				need_update_.insert(it->second.begin(), it->second.end());
			}
		}
	}

private:
	void add_caches (ade::TensT& roots)
	{
		ade::GraphStat stat;
		for (auto& root : roots)
		{
			root->accept(stat);
		}
		llo::iVariable* var;
		std::unordered_map<llo::iVariable*,ade::PathFinder> variables;
		for (auto& gpair : stat.graphsize_)
		{
			if (gpair.second == 0 &&
				(var = dynamic_cast<llo::iVariable*>(gpair.first)))
			{
				if (variables.end() == variables.find(var))
				{
					variables.emplace(var, ade::PathFinder(var));
				}
			}
		}

		for (auto& root : roots)
		{
			for (auto& varpair : variables)
			{
				root->accept(varpair.second);
			}
		}

		// calculate path intersections
		std::unordered_map<ade::iFunctor*,
			std::unordered_set<llo::iVariable*>> descendents_;
		for (auto& varpair : variables)
		{
			for (auto ppair : varpair.second.parents_)
			{
				auto f = static_cast<ade::iFunctor*>(ppair.first);
				descendents_[f].emplace(varpair.first);
			}
		}

		// add to cache buckets for functors with more than a single descendent
		for (auto it = descendents_.begin(), et = descendents_.end();
			it != et; ++it)
		{
			if (1 < it->second.size() && it->first->get_children().size() > 1)
			{
				if (caches_.end() == caches_.find(it->first))
				{
					caches_.emplace(it->first, nullptr);
				}
				for (llo::iVariable* var : it->second)
				{
					ancestors_[var].emplace(it->first);
				}
			}
		}
	}

	std::unordered_set<ade::iFunctor*> need_update_;

	// hide caches_ to ensure target consistency
	std::unordered_map<ade::iFunctor*,TensptrT<T>> caches_;

	std::unordered_map<llo::iVariable*,
		std::unordered_set<ade::iFunctor*>> ancestors_;

};

}

#endif // LLO_CACHE_HPP
