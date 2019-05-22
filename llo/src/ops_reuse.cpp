#include <list>

#include "llo/opt/ops_merge.hpp"
#include "llo/opt/ops_reuse.hpp"

#include "llo/constant.hpp"

#ifdef LLO_OPS_REUSE_HPP

namespace llo
{

static bool const_equals (Constant* lhs, Constant* rhs)
{
	age::_GENERATED_DTYPE dtype = (age::_GENERATED_DTYPE) lhs->type_code();
	if (dtype != rhs->type_code())
	{
		return false;
	}

	auto& lhs_shape = lhs->shape();
	auto& rhs_shape = rhs->shape();
	if (false == std::equal(lhs_shape.begin(),
		lhs_shape.end(), rhs_shape.begin()))
	{
		return false;
	}

	char* lhs_ptr = (char*) lhs->data();
	char* rhs_ptr = (char*) rhs->data();
	return std::equal(lhs_ptr, lhs_ptr +
		lhs_shape.n_elems() * age::type_size(dtype), rhs_ptr);
}

static bool coorder_equal (ade::CoordptrT lhs, ade::CoordptrT rhs)
{
	if (lhs == rhs)
	{
		return true;
	}
	bool is_equal = true;
	lhs->access(
	[&rhs, &is_equal](const ade::MatrixT& lhs_m)
	{
		rhs->access(
		[&lhs_m, &is_equal](const ade::MatrixT& rhs_m)
		{
			for (uint8_t i = 0; i < ade::mat_dim; ++i)
			{
				for (uint8_t j = 0; j < ade::mat_dim; ++j)
				{
					is_equal = is_equal && lhs_m[i][j] == rhs_m[i][j];
				}
			}
		});
	});
	return is_equal;
}

static bool func_equals (ade::iFunctor* lhs, ade::iFunctor* rhs,
	const std::unordered_map<ade::iTensor*,ade::iTensor*>& orig2new)
{
	auto& lhs_shape = lhs->shape();
	auto& rhs_shape = rhs->shape();
	if (false == std::equal(lhs_shape.begin(),
		lhs_shape.end(), rhs_shape.begin()))
	{
		return false;
	}

	auto et = orig2new.end();
	auto& lhs_children = lhs->get_children();
	auto& rhs_children = rhs->get_children();
	auto arg_equal =
		[&orig2new,&et](
			const ade::FuncArg& lhs,
			const ade::FuncArg& rhs)
		{
			auto lhs_tens = lhs.get_tensor().get();
			auto rhs_tens = rhs.get_tensor().get();
			auto lit = orig2new.find(lhs_tens);
			auto rit = orig2new.find(rhs_tens);
			if (lit != et)
			{
				lhs_tens = lit->second;
			}
			if (rit != et)
			{
				rhs_tens = rit->second;
			}
			return lhs_tens == rhs_tens &&
				coorder_equal(lhs.get_coorder(), rhs.get_coorder());
		};
	if (nnary_codes.end() == nnary_codes.find(
		(age::_GENERATED_OPCODE) lhs->get_opcode().code_))
	{
		// order matters
		// child check is expensive so check size before equality
		return std::equal(lhs_children.begin(), lhs_children.end(),
			rhs_children.begin(), arg_equal);
	}
	size_t nchildren = lhs_children.size();
	if (nchildren != rhs_children.size())
	{
		return false;
	}
	auto rit = rhs_children.begin();
	auto ret = rhs_children.end();
	return std::all_of(lhs_children.begin(), lhs_children.end(),
		[&](const ade::FuncArg& lhs_child)
		{
			return std::any_of(rit, ret,
				[&](const ade::FuncArg& rhs_child)
				{
					return arg_equal(lhs_child, rhs_child);
				});
		});
}

ade::TensT ops_reuse (ade::TensT roots)
{
	std::unordered_map<ade::iTensor*,ade::TensptrT> smart_map;
	ade::GraphStat stat;
	for (ade::TensptrT& root : roots)
	{
		smart_map.emplace(root.get(), root);
		root->accept(stat);
	}
	if (stat.graphsize_.size() == 0)
	{
		return roots;
	}

	size_t max_graphsize = 0;
	for (ade::TensptrT& root : roots)
	{
		max_graphsize = std::max(max_graphsize,
			stat.graphsize_[root.get()].upper_ + 1);
	}

	std::vector<std::list<ade::iTensor*>> tens(max_graphsize);
	for (std::pair<ade::iTensor*,ade::NumRange<size_t>> graphpair :
		stat.graphsize_)
	{
		ade::iTensor* ten = graphpair.first;
		size_t index = graphpair.second.upper_;
		if (smart_map.end() != smart_map.find(ten))
		{
			tens[index].push_front(ten);
		}
		else
		{
			tens[index].push_back(ten);
		}
	}

	// assert stat.graphsize_.size() > 0, hence tens.size() > 0
	std::unordered_map<ade::iTensor*,ade::iTensor*> orig2new;
	{
		std::unordered_map<size_t,std::list<Constant*>> hashs;
		for (ade::iTensor* leaf : tens[0])
		{
			if (auto cst = dynamic_cast<Constant*>(leaf))
			{
				bool not_found = true;
				auto& shape = cst->shape();
				size_t hashidx = std::hash<std::string>()(
					std::string(shape.begin(), shape.end()));
				auto& potential_eqs = hashs[hashidx];
				for (Constant* potential_eq : potential_eqs)
				{
					if (const_equals(cst, potential_eq))
					{
						orig2new.emplace(cst, potential_eq);
						not_found = false;
						break;
					}
				}
				if (not_found)
				{
					potential_eqs.push_back(cst);
				}
			}
		}
	}

	for (size_t i = 1, n = tens.size(); i < n; ++i)
	{
		std::unordered_map<size_t,std::list<ade::iFunctor*>> hashs;
		for (ade::iTensor* ten : tens[i])
		{
			bool not_found = true;
			auto func = static_cast<ade::iFunctor*>(ten);
			// populate smart map
			auto& children = func->get_children();
			for (auto& child : children)
			{
				auto smart = child.get_tensor();
				smart_map[smart.get()] = smart;
			}

			// find equalities
			size_t hashidx = func->get_opcode().code_;
			auto& potential_eqs = hashs[hashidx];
			for (ade::iFunctor* potential_eq : potential_eqs)
			{
				if (func_equals(func, potential_eq, orig2new))
				{
					orig2new.emplace(func, potential_eq);
					not_found = false;
					break;
				}
			}
			if (not_found)
			{
				potential_eqs.push_back(func);
			}
		}
	}
	std::unordered_map<ade::iTensor*,std::vector<ade::iTensor*>> new2origs;
	for (auto& replace_pair : orig2new)
	{
		new2origs[replace_pair.second].push_back(replace_pair.first);
	}

	for (size_t i = 1, n = tens.size(); i < n; ++i)
	{
		for (ade::iTensor* ten : tens[i])
		{
			// only update functors that are not replaced
			if (orig2new.end() == orig2new.find(ten))
			{
				auto func = static_cast<ade::iFunctor*>(ten);
				bool changed = false;
				ade::ArgsT children = func->get_children();
				for (size_t i = 0, n = children.size(); i < n; ++i)
				{
					auto it = orig2new.find(children[i].get_tensor().get());
					if (orig2new.end() != it)
					{
						changed = true;
						children[i] = ade::FuncArg
						{
							smart_map[it->second],
							children[i].get_shaper(),
							children[i].map_io(),
							children[i].get_coorder(),
						};
					}
				}
				if (changed)
				{
					auto f = ade::Functor::get(func->get_opcode(), children);
					ade::TensptrT optimized(f);
					// update smart and orig2new
					smart_map.emplace(f, optimized);
					auto it = new2origs.find(func);
					if (new2origs.end() != it)
					{
						// reference new updated functor instead of old one
						for (ade::iTensor* orig : it->second)
						{
							orig2new[orig] = f;
						}
					}
					orig2new[func] = f;
					new2origs[f].push_back(func);
				}
			}
		}
	}

	for (auto& root : roots)
	{
		auto rit = orig2new.find(root.get());
		if (orig2new.end() != rit)
		{
			root = smart_map[rit->second];
		}
	}
	return roots;
}

}

#endif
