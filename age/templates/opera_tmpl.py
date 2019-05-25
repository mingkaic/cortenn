''' Representation of operation mapping files '''

import age.templates.template as template

FILENAME = 'opmap'

_common_in = '_GENERATED_OPCODE opcode, ade::Shape shape, {data_in} in)'

_ref_signature = 'void typed_exec ({data_out} out, ' + _common_in

_ret_signature = '{data_out} typed_exec (' + _common_in

def _parse_signature(data):
    data_out = data['out']
    data_in = data['in']
    fmt = _ref_signature
    if isinstance(data_out, dict):
        if 'return' in data_out and data_out['return']:
            fmt = _ret_signature
        data_out = data_out['type']
    return fmt.format(data_out=data_out, data_in=data_in)

def _defval_stmt(data):
    data_out = data['out']
    if isinstance(data_out, dict) and \
        'return' in data_out and data_out['return']:
        stmt = '\n    {} defval;\n    return defval;'\
            .format(data_out['type'])
    else:
        stmt = ''
    return stmt

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_OPERA_HPP
#define _GENERATED_OPERA_HPP

namespace age
{{

template <typename T>
{signature}
{{
    switch (opcode)
    {{
{ops}
        default: logs::fatal("unknown opcode");
    }}{defreturn}
}}

// GENERIC_MACRO must accept a real type as an argument.
// e.g.:
// #define GENERIC_MACRO(REAL_TYPE) run<REAL_TYPE>(args...);
// ...
// TYPE_LOOKUP(GENERIC_MACRO, type_code)
#define TYPE_LOOKUP(GENERIC_MACRO, DTYPE)\\
switch (DTYPE) {{\\
{generic_macros}\\
    default: logs::fatal("executing bad type");\\
}}

}}

#endif // _GENERATED_OPERA_HPP
''')

header.signature = ('signatures.data', _parse_signature)

header.ops = ('opcodes', lambda opcodes: '\n'.join(['''        case {code}:
            {retval}; break;'''.format(\
    code = code, retval = opcodes[code]['operation']) for code in template.sortkey(opcodes)]))

header.generic_macros = ('dtypes', lambda dtypes: '\\\n'.join([
    '    case age::{dtype}: GENERIC_MACRO({real_type}) break;'.format(\
    dtype = dtype, real_type = dtypes[dtype]) for dtype in template.sortkey(dtypes)]))

header.defreturn = ('signatures.data', _defval_stmt)
