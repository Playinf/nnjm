/* parameter.cpp */
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <parameter.h>

namespace infinity {
namespace lm {

parameter::parameter()
{
    // model specific
    add_parameter("model", type::string);
    add_parameter("order", type::integer);
    add_parameter("window", type::integer);
    add_parameter("source-context", type::integer);
    add_parameter("target-context", type::integer);

    // neural network architecture
    add_parameter("network", type::string);
    add_parameter("layer-number", type::integer);
    add_parameter("feature-number", type::integer);
    add_parameter("input-size", type::integer);
    add_parameter("hidden-size", type::integer | type::vector);
    add_parameter("output-size", type::integer);
    add_parameter("activation-function", type::string | type::vector);
    add_parameter("output-function", type::string);

    // training options
    add_parameter("learning-rate", type::real);
    add_parameter("weight-decay", type::real);
    add_parameter("momentum", type::real | type::vector);
    add_parameter("norm-control", type::real);
    add_parameter("epoch", type::integer);
    add_parameter("max-epoch", type::integer);
    add_parameter("batch-size", type::integer);
    add_parameter("init-method", type::string);
    add_parameter("init-range", type::real | type::vector);
    add_parameter("cost-function", type::string);
    add_parameter("update-range", type::real | type::vector);
    add_parameter("weight-range", type::real | type::vector);

    // file specific options
    add_parameter("training-file", type::string);
    add_parameter("testing-file", type::string);
    add_parameter("validation-file", type::string);
    add_parameter("model-file", type::string);
    add_parameter("input-vocab", type::string | type::vector);
    add_parameter("output-vocab", type::string);

    // for debug
    add_parameter("verbose", type::integer);

    // set parameter flag
    set_parameter_flag("learning-rate", flag::overwrite);
    set_parameter_flag("weight-decay", flag::overwrite);
    set_parameter_flag("momentum", flag::overwrite);
    set_parameter_flag("norm-control", flag::overwrite);
    set_parameter_flag("max-epoch", flag::overwrite);
    set_parameter_flag("update-range", flag::overwrite);
    set_parameter_flag("weight-range", flag::overwrite);
    set_parameter_flag("batch-size", flag::overwrite);
    set_parameter_flag("training-file", flag::overwrite);
    set_parameter_flag("testing-file", flag::overwrite);
    set_parameter_flag("validation-file", flag::overwrite);
    set_parameter_flag("model-file", flag::overwrite);
    set_parameter_flag("input-vocab", flag::overwrite);
    set_parameter_flag("output-vocab", flag::overwrite);
    set_parameter_flag("verbose", flag::overwrite);

    // set parameters
    set_parameter("model", "nnlm");
    set_parameter("order", 0u);
    set_parameter("window", 0u);
    set_parameter("source-context", 0u);
    set_parameter("target-context", 0u);

    set_parameter("network", "ffnn");
    set_parameter("layer-number", 3u);
    set_parameter("feature-number", 100u);
    set_parameter("input-size", 0u);
    set_parameter("output-size", 0u);
    set_parameter("output-function", "identity");

    set_parameter("learning-rate", 1.0);
    set_parameter("weight-decay", 0.0);
    set_parameter("norm-control", 0.0);
    set_parameter("max-epoch", 50u);
    set_parameter("epoch", 0u);
    set_parameter("batch-size", 1u);
    set_parameter("init-method", "uniform");
    set_parameter("cost-function", "selfnorm");

    set_parameter("training-file", "");
    set_parameter("testing-file", "");
    set_parameter("validation-file", "");
    set_parameter("model-file", "");
    set_parameter("output-vocab", "");

    set_parameter("verbose", 0u);

    double inf = std::numeric_limits<double>::infinity();
    std::vector<double> hidden_size{ 0u };
    std::vector<std::string> act_func{ "identity", "tanh", "identity" };
    std::vector<double> momentum_vec{ 0.0, 0.0 };
    std::vector<double> init_range{ -1.0, 1.0 };
    std::vector<double> update_range { -inf, inf };
    std::vector<double> weight_range { -inf, inf };
    std::vector<std::string> input_vocab { "", "" };

    set_parameter("hidden-size", hidden_size);
    set_parameter("activation-function", act_func);
    set_parameter("momentum", momentum_vec);
    set_parameter("init-range", init_range);
    set_parameter("update-range", update_range);
    set_parameter("weight-range", weight_range);
    set_parameter("input-vocab", input_vocab);
}

parameter::~parameter()
{
    // do nothing
}

unsigned int parameter::get_parameter_number() const
{
    return parameter_map.size();
}

bool parameter::has_parameter(const std::string& n) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return false;

    return true;
}

int parameter::get_flag(const std::string& n, int& f) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto info = iter->second;
    f = info.flag;

    return 1;
}

int parameter::get_type(const std::string& n, int& t) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto info = iter->second;
    t = info.type;

    return 1;
}

int parameter::get_parameter(const std::string& n, double& v) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    if (info.type != type::real)
        return 0;

    v = info.value.double_value[0];

    return 1;
}

int parameter::get_parameter(const std::string& n, std::string& v) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    if (info.type != type::string)
        return 0;

    v = info.value.string_value[0];

    return 1;
}

int parameter::get_parameter(const std::string& n, unsigned int& v) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    if (info.type != type::integer)
        return 0;

    v = info.value.integer_value[0];

    return 1;
}

int parameter::get_parameter(const std::string& n, real_vector& v) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    if (info.type != (type::vector | type::real))
        return 0;

    v = info.value.double_value;

    return 1;
}

int parameter::get_parameter(const std::string& n, string_vector& v)
    const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    if (info.type != (type::vector | type::string))
        return 0;

    v = info.value.string_value;

    return 1;
}

int parameter::get_parameter(const std::string& n, integer_vector& v) const
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    if (info.type != (type::vector | type::integer))
        return 0;

    v = info.value.integer_value;

    return 1;
}

void parameter::add_parameter(const std::string& n, int t)
{
    auto& info = parameter_map[n];

    info.name = n;
    info.flag = flag::none;
    info.type = t;
}

void parameter::set_parameter_flag(const std::string& n, int f)
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return;

    auto& info = iter->second;

    info.flag = f;
}

void parameter::set_description(const std::string& n, const std::string& s)
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return;

    auto& info = iter->second;

    info.description = s;
}

int parameter::get_parameter(std::map<std::string, string_vector>& m)
    const
{
    for (auto& val : parameter_map) {
        auto& name = val.first;
        auto& info = val.second;
        int type = info.type;
        auto& vec = m[name];

        if (type & type::real) {
            unsigned int pnum = info.value.double_value.size();
            for (unsigned int i = 0; i < pnum; i++) {
                double v = info.value.double_value[i];
                vec.push_back(std::to_string(v));
            }
        } else if (type & type::string) {
            unsigned int pnum = info.value.string_value.size();
            for (unsigned int i = 0; i < pnum; i++) {
                vec.push_back(info.value.string_value[i]);
            }
        } else if (type & type::integer) {
            unsigned int pnum = info.value.integer_value.size();
            for (unsigned int i = 0; i < pnum; i++) {
                unsigned int v = info.value.integer_value[i];
                vec.push_back(std::to_string(v));
            }
        }
    }

    return 1;
}

int parameter::set_parameter(const std::string& n, double v)
{
    std::vector<double> vec{ v };

    return set_parameter(n, vec);
}

int parameter::set_parameter(const std::string& n, unsigned int v)
{
    std::vector<unsigned int> vec{ v };

    return set_parameter(n, vec);
}

int parameter::set_parameter(const std::string& n, const std::string& v)
{
    std::vector<std::string> vec{ v };

    return set_parameter(n, vec);
}

int parameter::set_parameter(const std::string& n, const real_vector& v)
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    info.value.double_value = v;

    return 1;
}

int parameter::set_parameter(const std::string& n, const string_vector& v)
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    info.value.string_value = v;

    return 1;
}

int parameter::set_parameter(const std::string& n, const integer_vector& v)
{
    auto iter = parameter_map.find(n);

    if (iter == parameter_map.end())
        return 0;

    auto& info = iter->second;

    info.value.integer_value = v;

    return 1;
}

} /* lm */
} /* infinity */
