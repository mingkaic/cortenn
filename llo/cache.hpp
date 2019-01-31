#include "ade/ade.hpp"

#include "llo/variable.hpp"

#ifndef LLO_CACHE_HPP
#define LLO_CACHE_HPP

namespace llo
{

// todo: investigate whether to reverse dependency between cache and evaluator
template <typename T>
struct CacheBucket final
{
	TensptrT<T> tens_;

	std::unordered_set<ade::iLeaf*> descendents_;
};

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
		auto it = caches_.find(key);
		if (caches_.end() != it)
		{
			it->second.tens_ = value;
		}
	}

	/// Return nullptr if key value needs updating
	TensptrT<T> get (ade::iFunctor* key) const
	{
		auto it = caches_.find(key);
		if (caches_.end() == it)
		{
			return nullptr;
		}
		// todo: add session info to determine whether to update key
		return it->second.tens_;
	}

	bool has_value (ade::iFunctor* key) const
	{
		return caches_.end() != caches_.find(key);
	}

	void add_equation (ade::TensT roots)
	{
		add_caches(roots);
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
		for (auto& gpair : stat.graphsize_)
		{
			if (gpair.second == 0 &&
				(var = dynamic_cast<llo::iVariable*>(gpair.first)))
			{
				if (variables_.end() == variables_.find(var))
				{
					variables_.emplace(var, ade::PathFinder(var));
				}
			}
		}

		for (auto& root : roots)
		{
			for (auto& varpair : variables_)
			{
				root->accept(varpair.second);
			}
		}

		// calculate path intersections
		for (auto& varpair : variables_)
		{
			for (auto ppair : varpair.second.parents_)
			{
				auto f = static_cast<ade::iFunctor*>(ppair.first);
				caches_[f].descendents_.emplace(varpair.first);
			}
		}

		// remove all cache buckets with a single descendent
		auto it = caches_.begin(), et = caches_.end();
		while (it != et)
		{
			if (2 > it.second.descendents_.size())
			{
				it = caches_.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	// hide caches_ to ensure target consistency
	std::unordered_map<ade::iFunctor*,CacheBucket<T>> caches_;

	std::unordered_map<llo::iVariable*,ade::PathFinder> variables_;
};

}

#endif // LLO_CACHE_HPP
