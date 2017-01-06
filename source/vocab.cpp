/* vocab.cpp */
#include <string>
#include <unordered_map>
#include <vocab.h>
#include <symbol.h>
#include <symtab.h>

namespace infinity {
namespace lm {

vocab::vocab()
{
    // do nothing
}

vocab::~vocab()
{
    // do nothing
}

vocab::iterator vocab::begin()
{
    return id_map.begin();
}

vocab::iterator vocab::end()
{
    return id_map.end();
}

vocab::const_iterator vocab::begin() const
{
    return id_map.begin();
}

vocab::const_iterator vocab::end() const
{
    return id_map.end();
}

vocab::iterator vocab::find(const std::string& s)
{
    return find(s, symbol_type::word);
}

vocab::iterator vocab::find(const std::string& s, symbol_type t)
{
    const symbol* sym;
    symtab* tab = symtab::get_instance();

    sym = tab->find_symbol(s, t);

    if (sym == nullptr)
        return end();

    return id_map.find(sym);
}

vocab::const_iterator vocab::find(const std::string& s) const
{
    return find(s, symbol_type::word);
}

vocab::const_iterator vocab::find(const std::string& s, symbol_type t) const
{
    const symbol* sym;
    symtab* tab = symtab::get_instance();

    sym = tab->find_symbol(s, t);

    if (sym == nullptr)
        return end();

    return id_map.find(sym);
}

unsigned int vocab::get_size() const
{
    return id_map.size();
}

unsigned int vocab::get_id(const char* s) const
{
    std::string str(s);

    return get_id(str);
}

unsigned int vocab::get_id(const std::string& s) const
{
    return get_id(s, symbol_type::word);
}

unsigned int vocab::get_id(const std::string& s, symbol_type t) const
{
    const symbol* sym;
    symtab* tab = symtab::get_instance();
    unsigned int not_found = static_cast<unsigned int>(-1);

    sym = tab->find_symbol(s, t);

    if (sym == nullptr)
        return not_found;

    auto iter = id_map.find(sym);

    if (iter == id_map.end())
        return not_found;

    return iter->second;
}

void vocab::insert(const std::string& name, unsigned int id)
{
    insert(name, symbol_type::word, id);
}

void vocab::insert(const std::string& name, symbol_type type, unsigned int id)
{
    const symbol* sym;
    symtab* tab = symtab::get_instance();

    sym = tab->search_symbol(name, type);
    id_map[sym] = id;
}

} /* lm */
} /* infinity */
