# AGE (ADE Generation Engine)

Generate glue api for ADE and some data manipulation library

# Configuration

The configuration script must be a json file in the following format:

{
    "includes": {
        "< output filename >": [
            "< c++ include path with quotes or triangle brackets >",
            ...
        ],
        ...
    },
    "dtypes": {
        "< type_code >": "< c++ type >",
        ...
    },
    "data": {
        "sum": "< op_code for nnary addition operation >",
        "prod": "< op_code for binary product operation >",
        "data_out": "< output type of op_exec >",
        "data_in": "< input type of op_exec >",
        "scalarize": "< call to create a tensor containing a double scalar filled to a specific shape >"
    },
    "opcodes": {
        "< op_code >": {
            "operation": "< operations call given (data_out out,ade::Shape shape,data_in in) signature >",
            "derivative": "< chain rule of opcode given  RuleSet::grad_rule signature >"
        },
        ...
    },
    "apis": [
        {
            "name": "< function name >",
            "args": [{
                "dtype": "< arg type >",
                "name": "< arg name >",
                "c": { // this is optional
                    "args": [{
                        "dtype": "< c arg type >",
                        "name": "< c arg name >"
                    }, ...],
                    "convert": "< combine args to c++ arg >"
                }
            }, ...],
            "out": "< output signature >"
        },
        ...
    ]
}

# Includepath

Generated files automatically add include paths to other generated files specified by --out directory + filename. --strip_prefix argument excludes the argument from the include path if the argument is the prefix
