''' Representation of gradient mapping files '''

import age.templates.template as template

FILENAME = 'grader'

def _decl_func(grad):
    grad_out = grad['out']
    grad_in = grad['in']
    fmt = '''{grad_out} chain_rule (ade::iFunctor* fwd,
    {grad_in} bwd, ade::TensT args, size_t idx);'''
    return fmt.format(grad_out=grad_out, grad_in=grad_in)

def _defn_func(grad):
    template = ''
    if 'template' in grad:
        template = grad['template']
        if len(template) > 0:
            template = 'template <{}>\n'.format(template)
    grad_out = grad['out']
    grad_in = grad['in']
    fmt = '''{template}{grad_out} chain_rule (ade::iFunctor* fwd,
    {grad_in} bwd, ade::TensT args, size_t idx)
{{
    switch (fwd->get_opcode().code_)
    {{
        _AGE_INTERNAL_GRADSWITCH()
        default: logs::fatal("no gradient rule for unknown opcode");
    }}
    {grad_out} defval;
    return defval;
}}'''
    return fmt.format(template=template, grad_out=grad_out, grad_in=grad_in)

def _decl_switch_gradop(opcodes):
    return '\\\n'.join([
        'case {code}: return {retval};'.format(
            code = code,
            retval = opcodes[code]['derivative'])
        for code in template.sortkey(opcodes)
    ])

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{{

#define _AGE_INTERNAL_GRADSWITCH()\\
{gradops}

{grad_decl}

}}

#endif // _GENERATED_GRADER_HPP
''')

header.gradops = ('opcodes', _decl_switch_gradop)

header.grad_decl = ('signatures.grad', lambda grad :
    _defn_func(grad) if 'template' in grad and
    len(grad['template']) > 0 else _decl_func(grad))

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_GRADER_HPP

namespace age
{{

{grad_defn}

}}

#endif
''')

source.grad_defn = ('signatures.grad', lambda grad:
    _defn_func(grad) if 'template' not in grad or
    len(grad['template']) == 0 else '')
