/* symtab.cpp */
#include <unordered_set>
#include <symbol.h>
#include <symtab.h>

namespace infinity {
namespace lm {

/* static symbol table instance */
symtab symtab::instance;

symtab::symtab()
{
    // do nothing
}

symtab::~symtab()
{
    // do nothing
}

unsigned int symtab::size() const
{
    return symbol_set.size();
}

const symbol* symtab::find_symbol(const std::string& s) const
{
    return find_symbol(s, symbol_type::word);
}

const symbol* symtab::find_symbol(const std::string& s, symbol_type t) const
{
    symbol sym(s, t);
    auto iter = symbol_set.find(sym);

    if (iter != symbol_set.end())
        return &(*iter);

    return nullptr;
}

const symbol* symtab::search_symbol(const std::string& s)
{
    return search_symbol(s, symbol_type::word);
}

const symbol* symtab::search_symbol(const std::string& s, symbol_type t)
{
    symbol sym(s, t);
    auto iter = symbol_set.find(sym);

    if (iter != symbol_set.end())
        return &(*iter);

    /* not in the symbol set */
    auto r1 = string_set.insert(s);
    symbol tmp(*r1.first, t);
    auto r2 = symbol_set.insert(tmp);

    return &(*r2.first);
}

symtab* symtab::get_instance()
{
    return &instance;
}

} /* lm */
} /* infinity */
