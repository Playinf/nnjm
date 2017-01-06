/*
 * symbol.h
 *
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __SYMBOL_H__
#define __SYMBOL_H__

#include <string>

namespace infinity {
namespace lm {

enum class symbol_type {
    none = 0,
    word = 1
};

class symbol {
public:

    symbol();
    symbol(const std::string& name);
    symbol(const std::string& name, symbol_type type);
    ~symbol();

    symbol& operator=(const symbol&) = delete;
    bool operator==(const symbol&) const;

    symbol_type get_type() const;
    const std::string* get_name() const;
private:
    symbol_type type;
    const std::string* name;
};

struct symbol_hash {
    std::size_t operator()(const symbol& sym) const;
};

struct symbol_equal {
    std::size_t operator()(const symbol& s1, const symbol& s2) const;
};

} /* lm */
} /* infinity */

#endif /* __SYMBOL_H__ */
