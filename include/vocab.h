/*
 * vocab.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __VOCAB_H__
#define __VOCAB_H__

#include <string>
#include <unordered_map>
#include <symbol.h>

namespace infinity {
namespace lm {

class vocab {
public:
    typedef std::unordered_map<const symbol*, unsigned int> map_type;
    typedef map_type::iterator iterator;
    typedef map_type::const_iterator const_iterator;

    vocab();
    ~vocab();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    iterator find(const std::string& s);
    iterator find(const std::string& s, symbol_type t);
    const_iterator find(const std::string& s) const;
    const_iterator find(const std::string& s, symbol_type t) const;

    unsigned int get_size() const;
    unsigned int get_id(const char* s) const;
    unsigned int get_id(const std::string& s) const;
    unsigned int get_id(const std::string& s, symbol_type t) const;

    void insert(const std::string& name, unsigned int id);
    void insert(const std::string& name, symbol_type t, unsigned int id);
private:
    map_type id_map;
};

} /* lm */
} /* infinity */

#endif /* __VOCAB_H__ */
