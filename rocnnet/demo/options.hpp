#include <chrono>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace fs = boost::filesystem;
namespace opt = boost::program_options;

struct Options
{
	Options (void) : desc_("Demo Options") {}

	bool parse(int argc, char** argv)
	{
		size_t default_seed = static_cast<size_t>(
			std::chrono::high_resolution_clock::now().
				time_since_epoch().count());

		std::vector<std::string> config_fnames;
		desc_.add_options()
			("help", "Display help message")
			("seed", opt::bool_switch(&seed_)->default_value(true), "whether to seed or not")
			("seedval", opt::value<size_t>(&seedval_)->default_value(default_seed),
				"number of times to test");

		opt::options_description all_options;
		all_options.add(desc_);

		try
		{
			opt::variables_map vars;
			opt::positional_options_description pos;
			opt::store(opt::command_line_parser(argc, argv).
				options(all_options).positional(pos).run(), vars);

			if (vars.count("help"))
			{
				std::cout << make_usage_string(
					fs::path(argv[0]).stem().string(), desc_, pos) << '\n';
				return false;
			}

			opt::notify(vars);
		}
		catch (opt::error& e)
		{
			std::cerr << "[ERROR]: " << e.what() << '\n';
			return false;
		}

		return true;
	}

	opt::options_description desc_;

	size_t n_train_;

	size_t n_test_;

	bool seed_;

	size_t seedval_;

private:
	std::string make_usage_string (const std::string& program_name,
		const opt::options_description& desc,
		opt::positional_options_description& pos)
	{
		std::stringstream usage;
		usage << "usage: ";
		usage << program_name << ' ';
		size_t N = pos.max_total_count();
		if(N == std::numeric_limits<unsigned>::max())
		{
			std::string arg;
			std::string last = pos.name_for_position(
				std::numeric_limits<unsigned>::max());
			for(size_t i = 0; arg != last; ++i)
			{
				arg = pos.name_for_position(i);
				usage << arg << ' ';
			}
			usage << '[' << arg << "] ";
			usage << "... ";
		}
		else
		{
			for(size_t i = 0; i < N; ++i)
			{
				usage << pos.name_for_position(i) << ' ';
			}
		}
		if(desc.options().size() > 0)
		{
			usage << "[options]";
		}
		usage << '\n' << desc;
		return usage.str();
	}
};
