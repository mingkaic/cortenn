''' Representation of operation mapping files '''

import age.templates.template as template

FILENAME = 'opmap'

def sortkey(dic):
    arr = dic.keys()
    arr.sort()
    return arr

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_OPERA_HPP
#define _GENERATED_OPERA_HPP

namespace age
{{

template <typename T>
void typed_exec (_GENERATED_OPCODE opcode,
    {data_out} out, ade::Shape shape, {data_in} in)
{{
    switch (opcode)
    {{
{ops}
        default: logs::fatal("unknown opcode");
    }}
}}

}}

#endif // _GENERATED_OPERA_HPP
''')

header.data_in = ('data.data_in', lambda data_in: data_in)

header.data_out = ('data.data_out', lambda data_out: data_out)

header.ops = ('opcodes', lambda opcodes: '\n'.join(['''        case {code}:
            {retval}; break;'''.format(\
    code = code, retval = opcodes[code]['operation']) for code in sortkey(opcodes)]))
