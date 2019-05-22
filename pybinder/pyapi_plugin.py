''' Extension to generate Pybind11 API file '''

import os

import age.templates.template as template

FILENAME = 'pyapi'

_DEFAULT_PYBIND_TYPE = 'double'

_PYBINDT = '<PybindT>'

_func_fmt = '''{outtype} {funcname}_{idx} ({param_decl})
{{
    return age::{funcname}({args});
}}'''

_template_prefixes = ['typename', 'class']

def _strip_template_prefix(template):
    for template_prefix in _template_prefixes:
        if template.startswith(template_prefix):
            return template[len(template_prefix):].strip()
    return template

def _wrap_func(idx, api):
    if 'template' in api and len(api['template']) > 0:
        templates = [_strip_template_prefix(typenames)
            for typenames in api['template'].split(',')]
    else:
        templates = []
    outtype = 'ade::TensptrT'
    if isinstance(api['out'], dict) and 'type' in api['out']:
        outtype = api['out']['type']

    out = _func_fmt.format(
        outtype=outtype,
        funcname = api['name'],
        idx = idx,
        param_decl = ', '.join([arg['dtype'] + ' ' + arg['name']
            for arg in api['args']]),
        args = ', '.join([arg['name'] for arg in api['args']]))
    for temp in templates:
        out = out.replace('<{}>'.format(temp), _PYBINDT)
    return out

_mdef_fmt = 'm.def("{pyfunc}", &pyage::{func}_{idx}, {description}, {pyargs});'

def _mdef_apis(apis):
    cnames = {}
    def checkpy(cname):
        if cname in cnames:
            out = cname + str(cnames[cname])
            cnames[cname] = cnames[cname] + 1
        else:
            out = cname
            cnames[cname] = 0
        return out

    def parse_header_args(arg):
        if 'default' in arg:
            defext = ' = {}'.format(arg['default'])
        else:
            defext = ''
        return '{dtype} {name}{defext}'.format(
            dtype = arg['dtype'],
            name = arg['name'],
            defext = defext)

    def parse_description(arg):
        if 'description' in arg:
            description = ': {}'.format(arg['description'])
        else:
            description = ''
        outtype = 'ade::TensptrT'
        if isinstance(arg['out'], dict) and 'type' in arg['out']:
            outtype = arg['out']['type']
        return '"{outtype} {func} ({args}){description}"'.format(
            outtype = outtype,
            func = arg['name'],
            args = ', '.join([parse_header_args(arg) for arg in arg['args']]),
            description = description)

    def parse_pyargs(arg):
        if 'default' in arg:
            defext = ' = {}'.format(arg['default'])
        else:
            defext = ''
        return 'py::arg("{name}"){defext}'.format(
            name = arg['name'],
            defext = defext)

    outtypes = set()
    for api in apis:
        if 'template' in api and len(api['template']) > 0:
            templates = [_strip_template_prefix(typenames)
                for typenames in api['template'].split(',')]
        else:
            templates = []
        if isinstance(api['out'], dict) and 'type' in api['out']:
            outtype = api['out']['type']
            for temp in templates:
                outtype = outtype.replace('<{}>'.format(temp), _PYBINDT)
            outtypes.add(outtype)

    class_defs = []
    for outtype in outtypes:
        if 'ade::TensptrT' == outtype:
            continue
        class_defs.append('py::class_<std::remove_reference<decltype(*{outtype}())>::type,{outtype}>(m, "{name}");'.format(
            outtype=outtype,
            name=outtype.split('::')[-1]))

    return '\n    '.join(class_defs) + '\n    ' +\
        '\n    '.join([_mdef_fmt.format(
        pyfunc = checkpy(api['name']),
        func = api['name'],
        idx = i,
        description = parse_description(api),
        pyargs = ', '.join([parse_pyargs(arg) for arg in api['args']]))
        for i, api in enumerate(apis)])

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''// type to replace template arguments in pybind
using PybindT = {pybind_type};
''')

header.pybind_type = ('pybind_type', lambda pybind_type: pybind_type or _DEFAULT_PYBIND_TYPE)

# EXPORT
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

source.unique_wrap = ('apis', lambda apis: '\n\n'.join([_wrap_func(i, api)
    for i, api in enumerate(apis)]))

source.defs = ('apis', _mdef_apis)

# EXPORT
def process(directory, relpath, fields):

    pybind_hdr_path = os.path.join(relpath, header.fpath)

    source.includes = [
        '"pybind11/pybind11.h"',
        '"pybind11/stl.h"',
        '"' + pybind_hdr_path + '"',
    ]

    directory['pyapi_hpp'] = header
    directory['pyapi_src'] = source

    header.process(fields)
    source.process(fields)

    if 'includes' in fields and source.fpath in fields['includes']:
        source.includes += fields['includes'][source.fpath]

    return directory
