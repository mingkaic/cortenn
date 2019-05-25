''' Representation of API files '''

import age.templates.template as template

FILENAME = 'api'

def _parse_args(arg, accept_def = True):
    if 'default' in arg and accept_def:
        defext = ' = {}'.format(arg['default'])
    else:
        defext = ''
    return '{dtype} {name}{defext}'.format(
        dtype = arg['dtype'],
        name = arg['name'],
        defext = defext)

def _decl_func(api):
    if 'description' in api:
        comment = '/**\n{}\n**/\n'.format(
            api['description'])
    else:
        comment = ''
    name = api['name']
    args = ', '.join([_parse_args(arg) for arg in api['args']])
    if isinstance(api['out'], dict) and 'type' in api['out']:
        outtype = api['out']['type']
    else:
        outtype = 'ade::TensptrT'

    return '{comment}{outtype} {api} ({args});'.format(
        comment = comment,
        outtype = outtype,
        api = name,
        args = args)

def _nullcheck(args):
    tens = list(filter(lambda arg: arg['dtype'] == 'ade::TensptrT', args))
    if len(tens) == 0:
        return 'false'
    varnames = [ten['name'] for ten in tens]
    return ' || '.join([varname + ' == nullptr' for varname in varnames])

def _defn_func(api):
    if 'template' in api:
        template = api['template']
    else:
        template = ''

    # treat as if header
    if len(template) > 0:
        template_prefix = 'template <{}>\n'.format(template)
        args = ', '.join([_parse_args(arg) for arg in api['args']])
    else:
        template_prefix = ''
        args = ', '.join([_parse_args(arg, accept_def=False)
            for arg in api['args']])
    outtype = 'ade::TensptrT'
    if isinstance(api['out'], dict):
        if 'type' in api['out']:
            outtype = api['out']['type']
        outval = api['out']['val']
    else:
        outval = api['out']

    return '''{template_prefix}{outtype} {api} ({args})
{{
    if ({null_check})
    {{
        logs::fatal("cannot {api} with a null argument");
    }}
    return {retval};
}}'''.format(
    template_prefix = template_prefix,
    outtype = outtype,
    api = api['name'],
    args = args,
    null_check = _nullcheck(api['args']),
    retval = outval)

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{{

{api_decls}

}}

#endif // _GENERATED_API_HPP
''')

header.api_decls = ('apis', lambda apis: '\n\n'.join([
    _defn_func(api) if 'template' in api and len(api['template']) > 0 else _decl_func(api)
    for api in apis]))

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_API_HPP

namespace age
{{

{apis}

}}

#endif
''')

source.apis = ('apis', lambda apis: '\n\n'.join([
    _defn_func(api) for api in apis
    if 'template' not in api or len(api['template']) == 0]))
