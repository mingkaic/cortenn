''' Representation of data conversion files '''

import age.templates.template as template

FILENAME = 'data'

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_DATA_HPP
#define _GENERATED_DATA_HPP

namespace age
{{

// uses std containers for type conversion
template <typename OUTTYPE>
void type_convert (std::vector<OUTTYPE>& out, void* input,
    age::_GENERATED_DTYPE intype, size_t nelems)
{{
    switch (intype)
    {{
{typed_conversions}
        default:
            logs::fatalf("invalid input type %s",
                age::name_type(intype).c_str());
    }}
}}

}}

#endif // _GENERATED_DATA_HPP
''')

header.typed_conversions = ('dtypes', lambda dtypes: '\n'.join([
    '''        case {dtype}:
            out = std::vector<OUTTYPE>(({real_type}*) input,
                ({real_type}*) input + nelems); break;'''.format(\
    dtype=dtype, real_type=dtypes[dtype]) for dtype in template.sortkey(dtypes)
]))
