/*
 * parameter.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <map>
#include <string>
#include <vector>

namespace infinity {
namespace lm {

class parameter {
public:
    class flag {
    public:
        static constexpr int none = 0;
        static constexpr int overwrite = 1;
    };

    class type {
    public:
        static constexpr int none = 0;
        static constexpr int real = 1;
        static constexpr int integer = 2;
        static constexpr int string = 4;
        static constexpr int vector = 8;
    };
public:
    typedef std::vector<double> real_vector;
    typedef std::vector<std::string> string_vector;
    typedef std::vector<unsigned int> integer_vector;

    parameter();
    ~parameter();

    parameter(const parameter&) = delete;
    parameter& operator=(const parameter&) = delete;

    unsigned int get_parameter_number() const;
    bool has_parameter(const std::string& n) const;
    int get_flag(const std::string& n, int& f) const;
    int get_type(const std::string& n, int& t) const;
    int get_parameter(const std::string& n, double& v) const;
    int get_parameter(const std::string& n, std::string& v) const;
    int get_parameter(const std::string& n, unsigned int& v) const;
    int get_parameter(const std::string& n, real_vector& v) const;
    int get_parameter(const std::string& n, string_vector& v) const;
    int get_parameter(const std::string& n, integer_vector& v) const;
    int get_parameter(std::map<std::string, string_vector>& m) const;

    int set_parameter(const std::string& n, double v);
    int set_parameter(const std::string& n, unsigned int v);
    int set_parameter(const std::string& n, const std::string& v);
    int set_parameter(const std::string& n, const real_vector& v);
    int set_parameter(const std::string& n, const string_vector& v);
    int set_parameter(const std::string& n, const integer_vector& v);
private:
    struct value_type {
        std::vector<double> double_value;
        std::vector<std::string> string_value;
        std::vector<unsigned int> integer_value;
    };

    struct information {
        int flag;
        int type;
        std::string name;
        value_type value;
        std::string description;
    };
private:
    void add_parameter(const std::string& n, int t);
    void set_parameter_flag(const std::string& n, int f);
    void set_description(const std::string& n, const std::string& s);
private:
    std::map<std::string, information> parameter_map;
};

} /* lm */
} /* infinity */

#endif /* __PARAMETER_H__ */
