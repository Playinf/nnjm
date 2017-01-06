/*
 * symtab.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __SYMTAB_H__
#define __SYMTAB_H__

#include <unordered_set>
#include <symbol.h>

namespace infinity {
namespace lm {

class symtab {
public:
    ~symtab();

    unsigned int size() const;
    const symbol* find_symbol(const std::string& s) const;
    const symbol* find_symbol(const std::string& s, symbol_type type) const;
    const symbol* search_symbol(const std::string& s);
    const symbol* search_symbol(const std::string& s, symbol_type type);
    static symtab* get_instance();
private:
    symtab();
private:
    std::unordered_set<std::string> string_set;
    std::unordered_set<symbol, symbol_hash, symbol_equal> symbol_set;
    static symtab instance;
};

} /* lm */
} /* infinity */

#endif // __SYMTAB_H__
