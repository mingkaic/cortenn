#include "ade/ileaf.hpp"
#include "bwd/grader.hpp"

#ifndef MOCK_GRADER_DEP_HPP
#define MOCK_GRADER_DEP_HPP

const size_t khaled_constant = 45;

struct MockTensor : public ade::iLeaf
{
	MockTensor (double scalar, ade::Shape shape) :
		scalar_(scalar), shape_(shape) {}

	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return "MockTensor";
	}

	void* data (void) override
	{
		return &scalar_;
	}

	const void* data (void) const override
	{
		return &scalar_;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	std::string type_label (void) const override
	{
		return "";
	}

	size_t nbytes (void) const override
	{
		return 0;
	}

	double scalar_;

	ade::Shape shape_;
};

ade::TensptrT arms_heavy (size_t idx, ade::TensT args);

ade::TensptrT dj_grad (ade::TensT args, size_t idx);

#endif // MOCK_GRADER_DEP_HPP
