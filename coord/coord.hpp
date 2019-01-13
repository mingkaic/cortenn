#include "Eigen/Core"
#include "Eigen/LU"

#include "ade/coord.hpp"

#ifndef COORD_COORDER_HPP
#define COORD_COORDER_HPP

namespace coord
{

struct EigenMap final : public ade::iCoordMap
{
	EigenMap (std::function<void(Eigen::MatrixXd& m)> init) :
		matrix_(ade::mat_dim, ade::mat_dim)
	{
		init(matrix_);
	}

	iCoordMap* connect (const iCoordMap& rhs) const override
	{
		if (auto emap = dynamic_cast<const EigenMap*>(&rhs))
		{
			return new EigenMap([this, &emap](Eigen::MatrixXd& m)
			{
				m = emap->matrix_ * this->matrix_;
			});
		}
		return new EigenMap([this, &rhs](Eigen::MatrixXd& m)
		{
			Eigen::MatrixXd otherm(ade::mat_dim, ade::mat_dim);
			rhs.access([&](const ade::MatrixT& om)
			{
				for (size_t i = 0; i < ade::mat_dim; ++i)
				{
					for (size_t j = 0; j < ade::mat_dim; ++j)
					{
						otherm(i, j) = om[i][j];
					}
				}
			});
			m = otherm * this->matrix_;
		});
	}

	void forward (ade::CoordT::iterator out,
		ade::CoordT::const_iterator in) const override
	{
		Eigen::VectorXd input(ade::mat_dim);
		for (size_t i = 0; i < ade::rank_cap; ++i)
		{
			input[i] = *(in + i);
		}
		input[ade::rank_cap] = 1;
		auto output = matrix_ * input;
		for (size_t i = 0; i < ade::rank_cap; ++i)
		{
			*(out + i) = output[i] / output[ade::rank_cap];
		}
	}

	iCoordMap* reverse (void) const override
	{
		return new EigenMap([this](Eigen::MatrixXd& m)
		{
			m = this->matrix_.inverse();
		});
	}

	std::string to_string (void) const override
	{
		std::stringstream ss;
		ss << matrix_;
		return ss.str();
	}

	void access (std::function<void(const ade::MatrixT&)> cb) const override
	{
		ade::MatrixT buffer;
		for (size_t i = 0; i < ade::mat_dim; ++i)
		{
			for (size_t j = 0; j < ade::mat_dim; ++j)
			{
				buffer[i][j] = matrix_(i, j);
			}
		}
		cb(buffer);
	}

	bool is_bijective (void) const override
	{
		return (int) matrix_.determinant() != 0;
	}

private:
	Eigen::MatrixXd matrix_;
};

/// Identity matrix instance
extern ade::CoordptrT identity;

/// Return coordinate mapper dividing dimensions after rank
/// by values in red vector
/// For example, given coordinate [2, 2, 6, 6], rank=2, and red=[3, 3],
/// mapper forward transforms to coordinate [2, 2, 2, 2]
ade::CoordptrT reduce (uint8_t rank, std::vector<ade::DimT> red);

/// Return coordinate mapper multiplying dimensions after rank
/// by values in ext vector
/// For example, given coordinate [6, 6, 2, 2], rank=2, and ext=[3, 3],
/// mapper forward transforms to coordinate [6, 6, 6, 6]
ade::CoordptrT extend (uint8_t rank, std::vector<ade::DimT> ext);

/// Return coordinate mapper permuting coordinate according to input order
/// Order is a vector of indices of the dimensions to appear in order
/// Indices not referenced by order but less than rank_cap will be appended
/// by numerical order
/// For example, given coordinate [1, 2, 3, 4], order=[1, 3],
/// mapper forward transforms to coordinate [2, 4, 1, 3]
/// Returned coordinate mapper will be a CoordMap instance, so inversibility
/// requires order indices be unique, otherwise throw fatal error
ade::CoordptrT permute (std::vector<uint8_t> order);

/// Return coordinate mapper flipping coordinate value at specified dimension
/// Flipped dimension with original value x is represented as -x-1
/// (see CoordT definition)
ade::CoordptrT flip (uint8_t dim);

}

#endif // COORD_COORDER_HPP
