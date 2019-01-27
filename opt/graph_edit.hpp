///
/// graph_editor.hpp
/// opt
///
/// Purpose:
/// Define ade graph pruning functions
///

#include "ade/ade.hpp"

#ifndef OPT_SHEAR_HPP
#define OPT_SHEAR_HPP

namespace opt
{

/// Edit functor type
using EditFuncT = std::function<ade::TensptrT(ade::Opcode,ade::ArgsT)>;

/// For some target extractable from iLeaf, prune graph such that reduces the
/// length of branches to target from root
/// For example, prune zeros branches by reducing f(x) * 0 to 0,
/// repeat for every instance of multiplication by zero in graph
/// Prune graph of root Tensptr
ade::TensptrT graph_edit (ade::TensptrT root, EditFuncT edit);

}

#endif // OPT_SHEAR_HPP
