#include "ade/ileaf.hpp"

#ifndef ADE_TEST_COMMON_HPP
#define ADE_TEST_COMMON_HPP

struct MockTensor final : public ade::iLeaf
{
	MockTensor (void) = default;

	MockTensor (ade::Shape shape) : shape_(shape) {}

	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

	void* data (void) override
	{
		return nullptr;
	}

	const void* data (void) const override
	{
		return nullptr;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	ade::Shape shape_;
};

#endif // ADE_TEST_COMMON_HPP
