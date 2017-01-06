/* constraint.cpp */
#include <limits>
#include <constraint.h>

namespace infinity {
namespace lm {

numeric_constraint::numeric_constraint()
{
    double inf = std::numeric_limits<double>::infinity();

    limit[0] = -inf;
    limit[1] = inf;
}

numeric_constraint::~numeric_constraint()
{
    // do nothing
}

double numeric_constraint::operator()(double v) const
{
    if (v < limit[0])
        return limit[0];
    else if (v > limit[1])
        return limit[1];
    else
        return v;
}

double numeric_constraint::get_lower_limit() const
{
    return limit[0];
}

double numeric_constraint::get_upper_limit() const
{
    return limit[1];
}

void numeric_constraint::set_limit(double l, double u)
{
    limit[0] = l;
    limit[1] = u;
}

} /* lm */
} /* infinity */
