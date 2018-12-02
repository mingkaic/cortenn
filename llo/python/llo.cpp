#include "llo/data.hpp"
#include "llo/zprune.hpp"

#include "llo/python/llo.hpp"

int64_t make_var (int shape[8], DTYPE dtype, char* label)
{
    age::_GENERATED_DTYPE gdtype;
    switch (dtype)
    {
        case INT:
            gdtype = age::INT32;
            break;
        case FLOAT:
            gdtype = age::DOUBLE;
            break;
        default:
            logs::fatal("cannot make variable of unknown type");
    }

    ade::Shape lshape(shape);
    llo::Variable* vp = new llo::Variable(nullptr, gdtype,
        lshape, label)

    return register_tens(vp);
}

void assign_int (int64_t var, int32_t* arr, int n)
{
    std::vector<int32_t> data(arr, n);
    auto vp = static_cast<llo::Variable*>(get_tens(var).get());
    *vp = data;
}

void assign_float (int64_t var, double* arr, int n)
{
    std::vector<double> data(arr, n);
    auto vp = static_cast<llo::Variable*>(get_tens(var).get());
    *vp = data;
}

void evaluate_int (int64_t root, int32_t* arr, int limit)
{
    auto tens = get_tens(root);
    llo::GenericData gdata = llo::eval(tens, age::INT32);
    int32_t* ptr = (int32_t*) gdata.data_.get();
    std::memcpy(arr, ptr,
        std::min(gdata.shape_.n_elems(), limit) * sizeof(int32_t));
}

void evaluate_float (int64_t root, double* arr, int limit)
{
    auto tens = get_tens(root);
    llo::GenericData gdata = llo::eval(tens, age::DOUBLE);
    double* ptr = (double*) gdata.data_.get();
    std::memcpy(arr, ptr,
        std::min(gdata.shape_.n_elems(), limit) * sizeof(double));
}

int64_t derive (int64_t root, int64_t wrt)
{
    auto rtens = get_tens(root);
    auto vtens = get_tens(wrt);
    ade::TensptrT der = llo::derive(rtens, vtens);
    return register_tens(der);
}
