''' Extension to generate Pybind11 API file '''

import os

import age.templates.template as template

FILENAME = 'pyapi'

source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''namespace py = pybind11;

namespace pyage
{{

{unique_wrap}

}}

PYBIND11_MODULE(age, m)
{{
	m.doc() = "pybind ade generator";

	py::class_<ade::iTensor,ade::TensptrT> tensor(m, "Tensor");

    {defs}
}}
''')

func_fmt = '''ade::TensptrT {funcname}_{idx} ({param_decl})
{{
    return age::{funcname}({args});
}}'''

mdef_fmt = 'm.def("{pyfunc}", &pyage::{func}_{idx}, "ade::TensptrT {func} ({param_decl})");'

def wrap_func(idx, api):
    return func_fmt.format(funcname=api['name'], idx=idx,
        param_decl=', '.join([arg['dtype'] + ' ' + arg['name'] for arg in api['args']]),
        args=', '.join([arg['name'] for arg in api['args']]))

def mdef_apis(apis):
    cnames = {}
    def checkpy(cname):
        if cname in cnames:
            out = cname + str(cnames[cname])
            cnames[cname] = cnames[cname] + 1
        else:
            out = cname
            cnames[cname] = 0
        return out
    return '\n    '.join([mdef_fmt.format(
        pyfunc=checkpy(api['name']), func=api['name'], idx=i,
        param_decl=', '.join([arg['dtype'] + ' ' + arg['name'] for arg in api['args']]))
        for i, api in enumerate(apis)])

source.unique_wrap = ('apis', lambda apis: '\n\n'.join([wrap_func(i, api)
    for i, api in enumerate(apis)]))

source.defs = ('apis', lambda apis: mdef_apis(apis))

def process(directory, relpath, fields):

    source.includes = [
        '"pybind11/pybind11.h"',
        '"pybind11/stl.h"',
        '"llo/generated/api.hpp"',
    ]

    directory['pyapi_src'] = source

    source.process(fields)

    return directory
