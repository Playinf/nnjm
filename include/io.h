/*
 * io.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __IO_H__
#define __IO_H__

namespace infinity {
namespace lm {

class model;
class vocab;

void save_model(const char* n, model* m);
void load_model(const char* n, model* m);
void load_vocab(const char* n, vocab* v);

} /* lm */
} /* infinity */

#endif /* __IO_H__ */
