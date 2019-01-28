''' Extension to generate Pybind11 API file '''

import age.templates.template as template

FILENAME = 'convert'

header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_CONV_CODES_HPP
#define _GENERATED_CONV_CODES_HPP

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

#endif // _GENERATED_CONV_CODES_HPP
''')

conversion_fmt = '''
		case age::{code}:
			out = std::vector<OUTTYPE>(
                ({real_type}*) input, ({real_type}*) input + nelems);
			break;'''

header.typed_conversions = {'dtypes', lambda dtypes: '\n'.join([
    conversion_fmt.format(code=code, real_type=dtypes[code]) for code in dtypes
])}

def process(directory, relpath, fields):

    header.includes = ['"llo/generated/codes.hpp"']

    directory['convert_hpp'] = header

    header.process(fields)

    return directory
