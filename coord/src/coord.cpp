#include "coord/coord.hpp"

#ifdef COORD_COORDER_HPP

namespace coord
{

ade::CoordptrT identity(new EigenMap(
	[](Eigen::MatrixXd& fwd)
	{
		for (uint8_t i = 0; i < ade::rank_cap; ++i)
		{
			fwd(i, i) = 1;
		}
	}));

ade::CoordptrT reduce (uint8_t rank, std::vector<ade::DimT> red)
{
	uint8_t n_red = red.size();
	if (std::any_of(red.begin(), red.end(),
		[](ade::DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot reduce using zero dimensions %s",
			fmts::to_string(red.begin(), red.end()).c_str());
	}
	if (rank + n_red > ade::rank_cap)
	{
		logs::fatalf("cannot reduce shape rank %d beyond rank_cap with n_red %d",
			rank, n_red);
	}
	if (0 == n_red)
	{
		logs::warn("reducing with empty vector ... will do nothing");
		return identity;
	}

	return ade::CoordptrT(new EigenMap(
		[&](Eigen::MatrixXd& fwd)
		{
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd(i, i) = 1;
			}
			for (uint8_t i = 0; i < n_red; ++i)
			{
				uint8_t outi = rank + i;
				fwd(outi, outi) = 1.0 / red[i];
			}
		}));
}

ade::CoordptrT extend (uint8_t rank, std::vector<ade::DimT> ext)
{
	uint8_t n_ext = ext.size();
	if (std::any_of(ext.begin(), ext.end(),
		[](ade::DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot extend using zero dimensions %s",
			fmts::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > ade::rank_cap)
	{
		logs::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		logs::warn("extending with empty vector ... will do nothing");
		return identity;
	}

	return ade::CoordptrT(new EigenMap(
		[&](Eigen::MatrixXd& fwd)
		{
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd(i, i) = 1;
			}
			for (uint8_t i = 0; i < n_ext; ++i)
			{
				uint8_t outi = rank + i;
				fwd(outi, outi) = ext[i];
			}
		}));
}

ade::CoordptrT permute (std::vector<uint8_t> dims)
{
	if (dims.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return identity;
	}

	bool visited[ade::rank_cap];
	std::memset(visited, false, ade::rank_cap);
	for (uint8_t i = 0, n = dims.size(); i < n; ++i)
	{
		visited[dims[i]] = true;
	}
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	return ade::CoordptrT(new EigenMap(
		[&](Eigen::MatrixXd& fwd)
		{
			for (uint8_t i = 0, n = dims.size(); i < n; ++i)
			{
				fwd(dims[i], i) = 1;
			}
		}));
}

ade::CoordptrT flip (uint8_t dim)
{
	if (dim >= ade::rank_cap)
	{
		logs::warn("flipping dimension out of rank_cap ... will do nothing");
		return identity;
	}

	return ade::CoordptrT(new EigenMap(
		[&](Eigen::MatrixXd& fwd)
		{
			for (uint8_t i = 0; i < ade::rank_cap; ++i)
			{
				fwd(i, i) = 1;
			}
			fwd(dim, dim) = -1;
			fwd(ade::rank_cap, dim) = -1;
		}));
}

}

#endif
