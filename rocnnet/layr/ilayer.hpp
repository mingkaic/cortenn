#include <memory>

#include "llo/data.hpp"

#ifndef LAYR_ILAYER_HPP
#define LAYR_ILAYER_HPP

struct iLayer
{
	iLayer (std::string label) : label_(label) {}

	virtual ~iLayer (void) {}

	virtual std::vector<llo::VarptrT> get_variables (void) const = 0;

	std::string label_;
};

#endif // LAYR_ILAYER_HPP
