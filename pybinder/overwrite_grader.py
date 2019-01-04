''' Representation of gradient mapping files '''

import age.templates.template as template

FILENAME = 'grader'

def sortkey(dic):
    arr = dic.keys()
    arr.sort()
    return arr

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{{

template <typename T>
ade::LeafptrT data (T scalar, ade::Shape shape)
{{
    return {scalarize};
}}

struct RuleSet final : public iRuleSet
{{
    ade::LeafptrT data (double scalar, ade::Shape shape) override
    {{
        return age::data(scalar, shape);
    }}

    ade::Opcode sum_opcode (void) override
    {{
        return ade::Opcode{{"{sum}", {sum}}};
    }}

    ade::TensptrT chain_rule (ade::iFunctor* fwd,
        ade::MappedTensor bwd, ade::TensT args, size_t idx) override;
}};

}}

#endif // _GENERATED_GRADER_HPP
''')

header.scalarize = ('data.scalarize', lambda scalarize: scalarize)

header.sum = ('data.sum', lambda sum: sum)

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_GRADER_HPP

namespace age
{{

ade::TensptrT RuleSet::chain_rule (ade::iFunctor* fwd,
    ade::MappedTensor bwd, ade::TensT args, size_t idx)
{{
    switch (fwd->get_opcode().code_)
    {{
{gradops}
        default: logs::fatal("no gradient rule for unknown opcode");
    }}
}}

}}

#endif
''')

source.gradops = ('opcodes', lambda opcodes: '\n'.join([
    '        case {code}: return {retval};'.format(\
    code = code, retval = opcodes[code]['derivative']) for code in sortkey(opcodes)]))
