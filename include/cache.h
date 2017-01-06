/*
 * cache.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __CACHE_H__
#define __CACHE_H__

namespace infinity {
namespace lm {

class cache {
public:
    cache();
    ~cache();

    cache(const cache&) = delete;
    cache& operator=(const cache&) = delete;

    double get_hit_rate() const;

    void clear();
    void resize(unsigned int n);
    double* find(unsigned int* key, unsigned int n);
    void update(unsigned int* key, unsigned int n, double value);
    void update(unsigned int* key, unsigned int n, double* v, unsigned int m);
private:
    struct block {
        bool valid;
        double* data;
        unsigned int *tag;
        unsigned int tag_size;
        unsigned int data_size;
    };
private:
    block* memory;
    unsigned int count;
    unsigned int hit_count;
    unsigned int cache_size;
};

} /* lm */
} /* infinity */

#endif /* __CACHE_H__ */
