#include "dbg/ade_csv.hpp"

void multiline_replace (std::string& multiline)
{
	size_t i = 0;
	char nline = '\n';
	while ((i = multiline.find(nline, i)) != std::string::npos)
	{
		multiline.replace(i, 1, "\\");
	}
}

bool is_identity (ade::iCoordMap* coorder)
{
	if (ade::identity.get() == coorder)
	{
		return true;
	}
	bool id = true;
	coorder->access([&id](const ade::MatrixT& m)
	{
		for (uint8_t i = 0; id && i < ade::mat_dim; ++i)
		{
			for (uint8_t j = 0; id && j < ade::mat_dim; ++j)
			{
				id = id && m[i][j] == (i == j);
			}
		}
	});
	return id;
}
