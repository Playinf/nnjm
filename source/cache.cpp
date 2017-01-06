/* cache.cpp */
#include <cache.h>
#include <utility.h>

namespace infinity {
namespace lm {

cache::cache()
{
    count = 0;
    hit_count = 0;
    // default 16 MB cache
    cache_size = 16 * 1024 * 1024;
    memory = nullptr;
    resize(cache_size);
}

cache::~cache()
{
    clear();

    if (memory != nullptr)
        delete[] memory;
}

double cache::get_hit_rate() const
{
    return hit_count / (double) count;
}

void cache::clear()
{
    for (unsigned int i = 0; i < cache_size; i++) {
        block& b = memory[i];
        b.valid = false;
        b.tag_size = 0;
        b.data_size = 0;

        if (b.tag != nullptr) {
            delete[] b.tag;
            b.tag = nullptr;
        }

        if (b.data != nullptr) {
            delete[] b.data;
            b.data = nullptr;
        }
    }
}

void cache::resize(unsigned int n)
{
    cache_size = n;

    if (memory != nullptr)
        delete[] memory;

    memory = new block[cache_size];

    for (unsigned int i = 0; i < cache_size; i++) {
        block* b = &memory[i];
        b->tag_size = 0;
        b->data_size = 0;
        b->tag = nullptr;
        b->data = nullptr;
    }
}

double* cache::find(unsigned int* key, unsigned int n)
{
    std::size_t hashval = 0;

    count++;

    for (unsigned int i = 0; i < n; i++) {
        hashval = hash_combine(hashval, key[i]);
    }

    block& blk = memory[hashval % cache_size];

    if (!blk.valid || blk.tag_size != n)
        return nullptr;

    for (unsigned int i = 0; i < n; i++) {
        if (blk.tag[i] != key[i])
            return nullptr;
    }

    hit_count++;

    return blk.data;
}

void cache::update(unsigned int* key, unsigned int n, double value)
{
    update(key, n, &value, 1);
}

void cache::update(unsigned int* k, unsigned int n, double* v, unsigned int m)
{
    std::size_t hashval = 0;

    for (unsigned int i = 0; i < n; i++) {
        hashval = hash_combine(hashval, k[i]);
    }

    block* blk = &memory[hashval % cache_size];

    if (blk->tag != nullptr) {
        delete[] blk->tag;
    }

    blk->valid = true;
    blk->tag_size = n;
    blk->tag = new unsigned int[n];

    if (blk->data != nullptr)
        delete[] blk->data;

    blk->data_size = m;
    blk->data = new double[m];

    for (unsigned int i = 0; i < m; i++) {
        blk->data[i] = v[i];
    }

    for (unsigned int i = 0; i < n; i++) {
        blk->tag[i] = k[i];
    }
}

} /* lm */
} /* infinity */
