/* symbol.cpp */
#include <string>
#include <functional>
#include <unordered_set>
#include <symbol.h>

namespace infinity {
namespace lm {

symbol::symbol()
{
    name = nullptr;
    type = symbol_type::word;
}

symbol::symbol(const std::string& name)
{
    this->name = &name;
    this->type = symbol_type::word;
}

symbol::symbol(const std::string& name, symbol_type type)
{
    this->name = &name;
    this->type = type;
}

symbol::~symbol()
{
    /* do nothing */
}

const std::string* symbol::get_name() const
{
    return name;
}

symbol_type symbol::get_type() const
{
    return type;
}

bool symbol::operator==(const symbol& sym) const
{
    if (type == sym.type) {
        return name == sym.name;
    }

    return false;
}

std::size_t symbol_hash::operator()(const symbol& sym) const
{
    std::hash<std::string> hasher;

    return hasher(*sym.get_name());
}

std::size_t symbol_equal::operator()(const symbol& s1, const symbol& s2) const
{
    const std::string& str1 = *s1.get_name();
    const std::string& str2 = *s2.get_name();

    if (str1 != str2)
        return false;

    return s1.get_type() == s2.get_type();
}

} /* lm */
} /* infinity */
