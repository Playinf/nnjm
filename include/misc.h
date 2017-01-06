/*
 * misc.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __MISC_H__
#define __MISC_H__

#include <map>
#include <string>
#include <vector>

namespace infinity {
namespace lm {

typedef std::map<std::string, std::vector<std::string>> map_type;

class parameter;

void check_parameter(parameter* param);
void load_parameter(map_type& setting, parameter* param);
void overwrite_parameter(map_type& setting, parameter* param);

} /* lm */
} /* infinity */

#endif /* __MISC_H__ */
