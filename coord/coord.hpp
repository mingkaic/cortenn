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

	bool is_bijective (void) const
	{
		return (int) matrix_.determinant() != 0;
	}

private:
	Eigen::MatrixXd matrix_;
};

}

#endif // COORD_COORDER_HPP
