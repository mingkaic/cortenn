import unittest
import difflib

import age.templates.api_tmpl as api
import age.templates.codes_tmpl as codes
import age.templates.data_tmpl as data
import age.templates.grader_tmpl as grader
import age.templates.opera_tmpl as opera

fields = {
    "opcodes": {
        "OP": {
            "operation": "foo(out, shape)",
            "derivative": "bwd_foo(args, idx)"
        },
        "OP1": {
            "operation": "bar1(out, shape, in)",
            "derivative": "bwd_bar(args[0], idx)"
        },
        "OP2": {
            "operation": "bar2(out, in[1])",
            "derivative": "foo(args[1])"
        },
        "OP3": {
            "operation": "bar2(out, in[0])",
            "derivative": "foo(args[0])"
        }
    },
    "dtypes": {
        "CAR": "char",
        "VROOM": "double",
        "VRUM": "float",
        "KAPOW": "complex_t"
    },
    "signatures": {
        "data": {
            "in": "In_Type",
            "out": "Out_Type"
        },
        "grad": {
            "out": "ade::TensptrT",
            "in": "ade::FuncArg"
        }
    },
    "apis": [
        {"name": "func1", "args": [], "out": "bar1()"},
        {"name": "func2", "args": [{
            "dtype": "ade::TensptrT",
            "name": "arg"
        }, {
            "dtype": "Arg",
            "name": "arg1",
            "c": {
                "args": [{
                    "dtype": "int",
                    "name": "n_arg1"
                }, {
                    "dtype": "float",
                    "name": "arg1_f"
                }],
                "convert": "Arg(arg1_f, n_arg1)"
            }
        }], "out": "bar2()"},
        {"name": "func3", "args": [{
            "dtype": "ade::TensptrT",
            "name": "arg"
        }, {
            "dtype": "Arg",
            "name": "arg1",
            "default": "Arg(2.4, 20)"
        }, {
            "dtype": "ade::TensptrT",
            "name": "arg2"
        }], "out": "bar3()"},
        {"name": "func1", "args": [{
            "dtype": "ade::TensT",
            "name": "arg"
        }, {
            "dtype": "Arg",
            "name": "arg1"
        }], "out": "bar4()",
        "description": "more complicated func1"}
    ]
}

api_header = """#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{

ade::TensptrT func1 ();

ade::TensptrT func2 (ade::TensptrT arg, Arg arg1);

ade::TensptrT func3 (ade::TensptrT arg, Arg arg1 = Arg(2.4, 20), ade::TensptrT arg2);

/**
more complicated func1
**/
ade::TensptrT func1 (ade::TensT arg, Arg arg1);

}

#endif // _GENERATED_API_HPP
"""

api_source = """#ifdef _GENERATED_API_HPP

namespace age
{

ade::TensptrT func1 ()
{
    if (false)
    {
        logs::fatal("cannot func1 with a null argument");
    }
    return bar1();
}

ade::TensptrT func2 (ade::TensptrT arg, Arg arg1)
{
    if (arg == nullptr)
    {
        logs::fatal("cannot func2 with a null argument");
    }
    return bar2();
}

ade::TensptrT func3 (ade::TensptrT arg, Arg arg1, ade::TensptrT arg2)
{
    if (arg == nullptr || arg2 == nullptr)
    {
        logs::fatal("cannot func3 with a null argument");
    }
    return bar3();
}

ade::TensptrT func1 (ade::TensT arg, Arg arg1)
{
    if (false)
    {
        logs::fatal("cannot func1 with a null argument");
    }
    return bar4();
}

}

#endif
"""

codes_header = """#ifndef _GENERATED_CODES_HPP
#define _GENERATED_CODES_HPP

namespace age
{

enum _GENERATED_OPCODE
{
    BAD_OP = 0,
    OP,
    OP1,
    OP2,
    OP3,
    _N_GENERATED_OPCODES,
};

enum _GENERATED_DTYPE
{
    BAD_TYPE = 0,
    CAR,
    KAPOW,
    VROOM,
    VRUM,
    _N_GENERATED_DTYPES,
};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (std::string name);

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (std::string name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{
    return BAD_TYPE;
}

template <>
_GENERATED_DTYPE get_type<char> (void);

template <>
_GENERATED_DTYPE get_type<complex_t> (void);

template <>
_GENERATED_DTYPE get_type<double> (void);

template <>
_GENERATED_DTYPE get_type<float> (void);

}

#endif // _GENERATED_CODES_HPP
"""

codes_source = """#ifdef _GENERATED_CODES_HPP

namespace age
{

struct EnumHash
{
    template <typename T>
    size_t operator() (T e) const
    {
        return static_cast<size_t>(e);
    }
};

static std::unordered_map<_GENERATED_OPCODE,std::string,EnumHash> code2name =
{
    { OP, "OP" },
    { OP1, "OP1" },
    { OP2, "OP2" },
    { OP3, "OP3" },
};

static std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{
    { "OP", OP },
    { "OP1", OP1 },
    { "OP2", OP2 },
    { "OP3", OP3 },
};

static std::unordered_map<_GENERATED_DTYPE,std::string,EnumHash> type2name =
{
    { CAR, "CAR" },
    { KAPOW, "KAPOW" },
    { VROOM, "VROOM" },
    { VRUM, "VRUM" },
};

static std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{
    { "CAR", CAR },
    { "KAPOW", KAPOW },
    { "VROOM", VROOM },
    { "VRUM", VRUM },
};

std::string name_op (_GENERATED_OPCODE code)
{
    auto it = code2name.find(code);
    if (code2name.end() == it)
    {
        return "BAD_OP";
    }
    return it->second;
}

_GENERATED_OPCODE get_op (std::string name)
{
    auto it = name2code.find(name);
    if (name2code.end() == it)
    {
        return BAD_OP;
    }
    return it->second;
}

std::string name_type (_GENERATED_DTYPE type)
{
    auto it = type2name.find(type);
    if (type2name.end() == it)
    {
        return "BAD_TYPE";
    }
    return it->second;
}

_GENERATED_DTYPE get_type (std::string name)
{
    auto it = name2type.find(name);
    if (name2type.end() == it)
    {
        return BAD_TYPE;
    }
    return it->second;
}

uint8_t type_size (_GENERATED_DTYPE type)
{
    switch (type)
    {
        case CAR: return sizeof(char);
        case KAPOW: return sizeof(complex_t);
        case VROOM: return sizeof(double);
        case VRUM: return sizeof(float);
        default: logs::fatal("cannot get size of bad type");
    }
    return 0;
}

template <>
_GENERATED_DTYPE get_type<char> (void)
{
    return CAR;
}

template <>
_GENERATED_DTYPE get_type<complex_t> (void)
{
    return KAPOW;
}

template <>
_GENERATED_DTYPE get_type<double> (void)
{
    return VROOM;
}

template <>
_GENERATED_DTYPE get_type<float> (void)
{
    return VRUM;
}

}

#endif
"""

data_header = """#ifndef _GENERATED_DATA_HPP
#define _GENERATED_DATA_HPP

namespace age
{

// uses std containers for type conversion
template <typename OUTTYPE>
void type_convert (std::vector<OUTTYPE>& out, void* input,
    age::_GENERATED_DTYPE intype, size_t nelems)
{
    switch (intype)
    {
        case CAR:
            out = std::vector<OUTTYPE>((char*) input,
                (char*) input + nelems); break;
        case KAPOW:
            out = std::vector<OUTTYPE>((complex_t*) input,
                (complex_t*) input + nelems); break;
        case VROOM:
            out = std::vector<OUTTYPE>((double*) input,
                (double*) input + nelems); break;
        case VRUM:
            out = std::vector<OUTTYPE>((float*) input,
                (float*) input + nelems); break;
        default:
            logs::fatalf("invalid input type %s",
                age::name_type(intype).c_str());
    }
}

}

#endif // _GENERATED_DATA_HPP
"""

grader_header = """#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{

#define _AGE_INTERNAL_GRADSWITCH()\\
case OP: return bwd_foo(args, idx);\\
case OP1: return bwd_bar(args[0], idx);\\
case OP2: return foo(args[1]);\\
case OP3: return foo(args[0]);

ade::TensptrT chain_rule (ade::iFunctor* fwd,
    ade::FuncArg bwd, ade::TensT args, size_t idx);

}

#endif // _GENERATED_GRADER_HPP
"""

grader_source = """#ifdef _GENERATED_GRADER_HPP

namespace age
{

ade::TensptrT chain_rule (ade::iFunctor* fwd,
    ade::FuncArg bwd, ade::TensT args, size_t idx)
{
    switch (fwd->get_opcode().code_)
    {
        _AGE_INTERNAL_GRADSWITCH()
        default: logs::fatal("no gradient rule for unknown opcode");
    }
    ade::TensptrT defval;
    return defval;
}

}

#endif
"""

opera_header = """#ifndef _GENERATED_OPERA_HPP
#define _GENERATED_OPERA_HPP

namespace age
{

template <typename T>
void typed_exec (Out_Type out, _GENERATED_OPCODE opcode, ade::Shape shape, In_Type in)
{
    switch (opcode)
    {
        case OP:
            foo(out, shape); break;
        case OP1:
            bar1(out, shape, in); break;
        case OP2:
            bar2(out, in[1]); break;
        case OP3:
            bar2(out, in[0]); break;
        default: logs::fatal("unknown opcode");
    }
}

// GENERIC_MACRO must accept a real type as an argument.
// e.g.:
// #define GENERIC_MACRO(REAL_TYPE) run<REAL_TYPE>(args...);
// ...
// TYPE_LOOKUP(GENERIC_MACRO, type_code)
#define TYPE_LOOKUP(GENERIC_MACRO, DTYPE)\\
switch (DTYPE) {\\
    case age::CAR: GENERIC_MACRO(char) break;\\
    case age::KAPOW: GENERIC_MACRO(complex_t) break;\\
    case age::VROOM: GENERIC_MACRO(double) break;\\
    case age::VRUM: GENERIC_MACRO(float) break;\\
    default: logs::fatal("executing bad type");\\
}

}

#endif // _GENERATED_OPERA_HPP
"""

def multiline_check(s1, s2):
    arr1 = list(filter(lambda line: len(line) > 0, s1.splitlines()))
    arr2 = list(filter(lambda line: len(line) > 0, s2.splitlines()))

    diffstr = '\n'.join(difflib.unified_diff(arr1, arr2))
    diffs = diffstr.splitlines()
    if len(diffs) > 0:
        print(diffstr)

class ClientTest(unittest.TestCase):
    def test_api(self):
        header = str(api.header.process(fields))
        source = str(api.source.process(fields))
        multiline_check(api_header, header)
        multiline_check(api_source, source)
        self.assertEqual(api_header, header)
        self.assertEqual(api_source, source)

    def test_codes(self):
        header = str(codes.header.process(fields))
        source = str(codes.source.process(fields))
        multiline_check(codes_header, header)
        multiline_check(codes_source, source)
        self.assertEqual(codes_header, header)
        self.assertEqual(codes_source, source)

    def test_data(self):
        header = str(data.header.process(fields))
        multiline_check(data_header, header)
        self.assertEqual(data_header, header)

    def test_grader(self):
        header = str(grader.header.process(fields))
        source = str(grader.source.process(fields))
        multiline_check(grader_header, header)
        multiline_check(grader_source, source)
        self.assertEqual(grader_header, header)
        self.assertEqual(grader_source, source)

    def test_opera(self):
        header = str(opera.header.process(fields))
        multiline_check(opera_header, header)
        self.assertEqual(opera_header, header)

if __name__ == "__main__":
    unittest.main()
