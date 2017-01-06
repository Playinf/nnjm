/*
 * constraint.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

namespace infinity {
namespace lm {

class numeric_constraint {
public:
    numeric_constraint();
    ~numeric_constraint();

    double operator()(double v) const;

    double get_lower_limit() const;
    double get_upper_limit() const;
    void set_limit(double l, double u);
private:
    double limit[2];
};

} /* lm */
} /* infinity */

#endif /* __CONSTRAINT_H__ */
