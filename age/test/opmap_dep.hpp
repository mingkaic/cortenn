#include "ade/shape.hpp"
#include "age/test/codes_dep.hpp"

#ifndef MOCK_OPMAP_DEP_HPP
#define MOCK_OPMAP_DEP_HPP

const size_t wraphash = 10;

const size_t djidx = 34;

struct Pomegranate
{
	static size_t meat_hash (Meat meat)
	{
		return meat.num + meat.size;
	}

	static size_t fries_hash (Fries fries)
	{
		return fries.num * fries.size;
	}

	void mix (Meat meat)
	{
		type_ = "meat";
		hash_ = meat_hash(meat);
	}

	void mix (Fries fries)
	{
		type_ = "fries";
		hash_ = fries_hash(fries);
	}

	ade::Shape shape_;
	std::string type_;
	size_t hash_;
};

struct SweetPotato
{
	Pomegranate* pptr_;
};

template <typename T>
void knees_weak (Pomegranate& in, ade::Shape shape, SweetPotato& out)
{
	in.mix(T(10));
	in.shape_ = shape;
	out.pptr_ = &in;
}

template <typename T>
void another_one (SweetPotato& out, Pomegranate& in, ade::Shape shape)
{
	in.mix(T(34));
	in.shape_ = shape;
	out.pptr_ = &in;
}

#endif // MOCK_OPMAP_DEP_HPP
