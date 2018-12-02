#include "llo/generated/capi.hpp"

extern enum DTYPE
{
    INT,
    FLOAT,
};

extern int64_t make_var (int shape[8], DTYPE dtype, char* label);

extern void assign_int (int64_t var, int32_t* arr, int n);

extern void assign_float (int64_t var, double* arr, int n);

extern void evaluate_int (int64_t root, int32_t* arr, int limit);

extern void evaluate_float (int64_t root, double* arr, int limit);

extern int64_t derive (int64_t root, int64_t wrt);
