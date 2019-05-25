''' Representation of OPCODE and DTYPE definition files '''

import age.templates.template as template

FILENAME = 'codes'

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_CODES_HPP
#define _GENERATED_CODES_HPP

namespace age
{{

enum _GENERATED_OPCODE
{{
    BAD_OP = 0,
{opcodes}
    _N_GENERATED_OPCODES,
}};

enum _GENERATED_DTYPE
{{
    BAD_TYPE = 0,
{dtypes}
    _N_GENERATED_DTYPES,
}};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (std::string name);

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (std::string name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{{
    return BAD_TYPE;
}}

{get_type_decls}

}}

#endif // _GENERATED_CODES_HPP
''')

header.opcodes = ('opcodes', lambda opcodes: '\n'.join(['    {code},'.format(\
    code = code) for code in template.sortkey(opcodes)]))

header.dtypes = ('dtypes', lambda dtypes: '\n'.join(['    {dtype},'.format(\
    dtype = dtype) for dtype in template.sortkey(dtypes)]))

header.get_type_decls = ('dtypes', lambda dtypes: '\n\n'.join(['''template <>
_GENERATED_DTYPE get_type<{real_type}> (void);'''.format(\
    real_type = dtypes[dtype]) for dtype in template.sortkey(dtypes)]))

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_CODES_HPP

namespace age
{{

struct EnumHash
{{
    template <typename T>
    size_t operator() (T e) const
    {{
        return static_cast<size_t>(e);
    }}
}};

static std::unordered_map<_GENERATED_OPCODE,std::string,EnumHash> code2name =
{{
{code2names}
}};

static std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{{
{name2codes}
}};

static std::unordered_map<_GENERATED_DTYPE,std::string,EnumHash> type2name =
{{
{type2names}
}};

static std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{{
{name2types}
}};

std::string name_op (_GENERATED_OPCODE code)
{{
    auto it = code2name.find(code);
    if (code2name.end() == it)
    {{
        return "BAD_OP";
    }}
    return it->second;
}}

_GENERATED_OPCODE get_op (std::string name)
{{
    auto it = name2code.find(name);
    if (name2code.end() == it)
    {{
        return BAD_OP;
    }}
    return it->second;
}}

std::string name_type (_GENERATED_DTYPE type)
{{
    auto it = type2name.find(type);
    if (type2name.end() == it)
    {{
        return "BAD_TYPE";
    }}
    return it->second;
}}

_GENERATED_DTYPE get_type (std::string name)
{{
    auto it = name2type.find(name);
    if (name2type.end() == it)
    {{
        return BAD_TYPE;
    }}
    return it->second;
}}

uint8_t type_size (_GENERATED_DTYPE type)
{{
    switch (type)
    {{
{type_sizes}
        default: logs::fatal("cannot get size of bad type");
    }}
    return 0;
}}

{get_types}

}}

#endif
''')

source.code2names = ('opcodes', lambda opcodes: '\n'.join(['    {{ {code}, "{code}" }},'.format(\
    code = code) for code in template.sortkey(opcodes)]))

source.name2codes = ('opcodes', lambda opcodes: '\n'.join(['    {{ "{code}", {code} }},'.format(\
    code = code) for code in template.sortkey(opcodes)]))

source.type2names = ('dtypes', lambda dtypes: '\n'.join(['    {{ {dtype}, "{dtype}" }},'.format(\
    dtype = dtype) for dtype in template.sortkey(dtypes)]))

source.name2types = ('dtypes', lambda dtypes: '\n'.join(['    {{ "{dtype}", {dtype} }},'.format(\
    dtype = dtype) for dtype in template.sortkey(dtypes)]))

source.type_sizes = ('dtypes', lambda dtypes: '\n'.join(['        case {dtype}: return sizeof({real_type});'.format(\
    dtype = dtype, real_type = dtypes[dtype]) for dtype in template.sortkey(dtypes)]))

source.get_types = ('dtypes', lambda dtypes: '\n\n'.join(['''template <>
_GENERATED_DTYPE get_type<{real_type}> (void)
{{
    return {dtype};
}}'''.format(dtype = dtype, real_type = dtypes[dtype]) for dtype in template.sortkey(dtypes)]))
