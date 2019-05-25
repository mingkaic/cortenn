#include <cassert>
#include "age/test/grader_dep.hpp"

ade::TensptrT arms_heavy (size_t idx, ade::TensT args)
{
	assert(args.size() > 0);
	static_cast<MockTensor*>(args[0].get())->scalar_ = idx;
	return args[0];
}

ade::TensptrT dj_grad (ade::TensT args, size_t idx)
{
	assert(args.size() > 0);
	static_cast<MockTensor*>(args[0].get())->scalar_ = idx + khaled_constant;
	return args[0];
}
