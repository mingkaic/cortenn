''' Extension to generate C API files '''

import os

import age.templates.template as template

from gen.plugin_base2 import PluginBase
from gen.file_rep import FileRep

_origtype = 'ade::TensptrT'
_repltype = 'int64_t'

_origarrtype = 'ade::TensT'

FILENAME = 'capi'

def affix_apis(apis):
    names = [api['name'] for api in apis]
    affix_maps = {name: names.count(name) - 1 for name in names}
    affixes = []
    for api in apis:
        nocc = affix_maps[api['name']]
        if nocc > 0:
            affix = '_' + str(nocc)
            affix_maps[api['name']] = nocc - 1
        else:
            affix = ''
        affixes.append((api, affix))
    return affixes

header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_CAPI_HPP
#define _GENERATED_CAPI_HPP

int64_t register_tens (ade::iTensor* ptr);

int64_t register_tens (ade::TensptrT& ptr);

ade::TensptrT get_tens (int64_t id);

extern void free_tens (int64_t id);

extern void get_shape (int outshape[8], int64_t tens);

{api_decls}

#endif // _GENERATED_CAPI_HPP
''')

_cfunc_sign_fmt = 'int64_t age_{ifunc} ({params})'

def _decl_func(api, affix):
    params = []
    for arg in api['args']:
        dtype = arg['dtype']
        argname = arg['name']
        if 'c' in arg:
            for cv in arg['c']['args']:
                params.append(cv['dtype'] + ' ' + cv['name'])
        elif dtype == _origtype:
            params.append(_repltype + ' ' + argname)
        elif dtype == _origarrtype:
            params.append('int64_t* ' + argname)
            params.append('uint64_t n_' + argname)
        else:
            params.append(dtype + ' ' + argname)
    return _cfunc_sign_fmt.format(ifunc = api['name'] + affix,
        params = ', '.join(params))

header.api_decls = ('apis', lambda apis: '\n\n'.join([\
    'extern ' + _decl_func(api, affix) + ';' for api, affix in affix_apis(apis)]))

source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_CAPI_HPP

static std::unordered_map<int64_t,ade::TensptrT> tens;

int64_t register_tens (ade::iTensor* ptr)
{{
    int64_t id = (int64_t) ptr;
    tens.emplace(id, ade::TensptrT(ptr));
    return id;
}}

int64_t register_tens (ade::TensptrT& ptr)
{{
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}}

ade::TensptrT get_tens (int64_t id)
{{
    auto it = tens.find(id);
    if (tens.end() == it)
    {{
        return ade::TensptrT(nullptr);
    }}
    return it->second;
}}

void free_tens (int64_t id)
{{
    tens.erase(id);
}}

void get_shape (int outshape[8], int64_t id)
{{
    const ade::Shape& shape = get_tens(id)->shape();
    std::copy(shape.begin(), shape.end(), outshape);
}}

{apis}

#endif
''')

_cfunc_bloc_fmt = '''{{{arg_decls}
    auto ptr = age::{func}({params});
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}}'''

def _defn_func(api, affix):
    decls = []
    params = []
    for arg in api['args']:
        dtype = arg['dtype']
        argname = arg['name']
        if 'c' in arg:
            params.append(arg['c']['convert'])
        elif dtype == _origtype:
            decls.append('ade::TensptrT {name}_ptr = get_tens({name});'
                .format(name=argname))
            params.append(argname + '_ptr')
        elif dtype == _origarrtype:
            decls.append('ade::TensT {name}_tens(n_{name});'.format(name=argname))
            decls.append('std::transform({name}, {name} + n_{name}, {name}_tens.begin(),'.\
                format(name=argname))
            decls.append('    [](int64_t id){ return get_tens(id); });')
            params.append(argname + '_tens')
        else:
            params.append(argname)
    arg_decls = '\n    '.join(decls)
    if len(arg_decls) > 0:
        arg_decls = '\n    ' + arg_decls
    return _decl_func(api, affix) + '\n' + _cfunc_bloc_fmt.format(
        arg_decls = arg_decls,
        func = api['name'],
        params = ', '.join(params))

source.apis = ('apis', lambda apis: '\n\n'.join([_defn_func(api, affix)\
    for api, affix in affix_apis(apis)]))

_plugin_id = 'CAPI'

class CAPIPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        generated_files[header.fpath] = FileRep(
            header.process(arguments),
            user_includes=[],
            internal_refs=[])

        generated_files[source.fpath] = FileRep(
            source.process(arguments),
            user_includes=[
                '<algorithm>',
                '<unordered_map>',
            ],
            internal_refs=[
                'api.hpp',
                header.fpath,
            ])

        if 'includes' in arguments and\
            header.fpath in arguments['includes']:
            generated_files[header.fpath] += arguments['includes'][header.fpath]

        if 'includes' in arguments and\
            source.fpath in arguments['includes']:
            generated_files[source.fpath] += arguments['includes'][source.fpath]

        return generated_files

PluginBase.register(CAPIPlugin)
