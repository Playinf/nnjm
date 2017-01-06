/*
 * hander.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __HANDLER_H__
#define __HANDLER_H__

namespace infinity {
namespace lm {

typedef double (*activation_handler)(double);
typedef double (*derivative_handler)(double);
typedef void (*output_handler)(double*, unsigned int);

} /* lm */
} /* infinity */

#endif /* __HANDLER_H__ */
