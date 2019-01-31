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
    const std::unordered_map<ade::iTensor*,ade::iTensor*>& replacements)
{
    auto& lhs_shape = lhs->shape();
    auto& rhs_shape = rhs->shape();
    if (false == std::equal(lhs_shape.begin(),
        lhs_shape.end(), rhs_shape.begin()))
    {
        return false;
    }

    auto et = replacements.end();
    auto& lhs_children = lhs->get_children();
    auto& rhs_children = rhs->get_children();
    if (nnary_codes.end() == nnary_codes.find(
        (age::_GENERATED_OPCODE) lhs->get_opcode().code_))
    {
        // order matters
        // child check is expensive so check size before equality
        return lhs_children.size() == rhs_children.size() &&
            std::equal(lhs_children.begin(), lhs_children.end(),
            rhs_children.begin(),
            [&replacements, &et](const ade::MappedTensor& lhs,
                const ade::MappedTensor& rhs)
            {
                auto lhs_tens = lhs.get_tensor().get();
                auto rhs_tens = rhs.get_tensor().get();
                auto et = replacements.end();
                auto lit = replacements.find(lhs_tens);
                auto rit = replacements.find(rhs_tens);
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
            });
    }
    // order doesn't matter
    std::unordered_map<ade::iTensor*,std::list<ade::CoordptrT>> lhs_map;
    for (const ade::MappedTensor& lhs_child : lhs_children)
    {
        auto lhs_tens = lhs_child.get_tensor().get();
        auto lit = replacements.find(lhs_tens);
        if (lit != et)
        {
            lhs_tens = lit->second;
        }
        lhs_map[lhs_tens].push_back(lhs_child.get_coorder());
    }
    for (const ade::MappedTensor& rhs_child : rhs_children)
    {
        auto rhs_tens = rhs_child.get_tensor().get();
        auto rit = replacements.find(rhs_tens);
        if (rit != et)
        {
            rhs_tens = rit->second;
        }
        auto it = lhs_map.find(rhs_tens);
        if (lhs_map.end() == it)
        {
            return false;
        }
        auto rhs_coord = rhs_child.get_coorder();
        auto cit = it->second.begin();
        auto cet = it->second.end();
        while (cit != cet && false == coorder_equal(*cit, rhs_coord))
        {
            ++cit;
        }
        if (cit == cet)
        {
            return false; // coordinates don't match
        }
        it->second.erase(cit);
    }
    return true;
}

ade::TensptrT ops_reuse (ade::TensptrT root)
{
	ade::GraphStat stat;
	root->accept(stat);
	if (stat.graphsize_.size() == 0)
	{
		return root;
	}

    std::vector<std::list<ade::iTensor*>> tens(stat.graphsize_[root.get()] + 1);
    for (std::pair<ade::iTensor*,size_t> graphpair : stat.graphsize_)
    {
        ade::iTensor* ten = graphpair.first;
        size_t index = graphpair.second;
        if (root.get() == ten)
        {
            tens[index].push_front(ten);
        }
        else
        {
            tens[index].push_back(ten);
        }
    }

    // assert stat.graphsize_.size() > 0, hence tens.size() > 0
	std::unordered_map<ade::iTensor*,ade::iTensor*> replacement;
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
                        replacement.emplace(cst, potential_eq);
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

    std::unordered_map<ade::iTensor*,ade::TensptrT> smart_map = {
        {root.get(), root},
    };
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
                if (func_equals(func, potential_eq, replacement))
                {
                    replacement.emplace(func, potential_eq);
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

    return opt::graph_edit(root, [&smart_map, replacement](bool& is_optimized,
	    ade::Opcode& opcode, ade::ArgsT& args) -> ade::TensptrT
        {
            auto et = replacement.end();
            for (size_t i = 0, n = args.size(); i < n; ++i)
            {
                auto it = replacement.find(args[i].get_tensor().get());
                if (et != it)
                {
                    is_optimized = true;
                    args[i] = ade::MappedTensor
                    {
                        smart_map[it->second],
                        args[i].get_shaper(),
                        args[i].map_io(),
                        args[i].get_coorder(),
                    };
                }
            }
            return nullptr;
        });
}

}

#endif
