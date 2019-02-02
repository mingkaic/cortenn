#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ade/ileaf.hpp"
#include "ade/ifunctor.hpp"

const char label_delim = ':';

void multiline_replace (std::string& multiline);

bool is_identity (ade::iCoordMap* coorder);

enum NODE_TYPE
{
	VARIABLE = 0,
	FUNCTOR,
	CACHED_FUNC,
};

using GetTypeFuncT = std::function<NODE_TYPE(ade::iFunctor*)>;

struct CSVEquation final : public ade::iTraveler
{
	CSVEquation (GetTypeFuncT get_ftype =
		[](ade::iFunctor* func) { return FUNCTOR; }) :
		get_ftype_(get_ftype) {}

	void visit (ade::iLeaf* leaf) override
	{
		nodes_.emplace(leaf, Node{
			leaf->to_string(),
			VARIABLE,
		});
	}

	void visit (ade::iFunctor* func) override
	{
		std::string funcstr = func->to_string();
		if (showshape_)
		{
			funcstr += func->shape().to_string();
		}
		nodes_.emplace(func, Node{
			funcstr,
			get_ftype_(func),
		});
		auto& children = func->get_children();
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			const ade::FuncArg& child = children[i];
			auto coorder = child.get_coorder().get();
			auto tens = child.get_tensor().get();
			if (is_identity(coorder))
			{
				coorder = nullptr;
			}
			else
			{
				std::string coordstr = coorder->to_string();
				multiline_replace(coordstr);
				coorders_.emplace(coorder, coordstr);
			}
			edges_.insert(Edge{
				func,
				tens,
				coorder,
				i
			});
			tens->accept(*this);
		}
	}

	void to_stream (std::ostream& out)
	{
		std::vector<ade::iTensor*> vecnodes(nodes_.size());
		std::transform(nodes_.begin(), nodes_.end(), vecnodes.begin(),
			[](std::pair<ade::iTensor*,Node> nodepair)
			{
				return nodepair.first;
			});

		std::unordered_map<ade::iTensor*,size_t> nodeidces;

		size_t nnodes = vecnodes.size();
		for (size_t i = 0; i < nnodes; ++i)
		{
			nodeidces.emplace(vecnodes[i], i);
		}

		size_t i = 0;
		for (auto it = edges_.begin(), et = edges_.end();
			it != et; ++it, ++i)
		{
			const Edge& edge = *it;
			std::string color = nodes_[edge.child_].ntype_ == CACHED_FUNC ?
				"red" : "white";
			if (nullptr == edge.coorder_)
			{
				out << nodeidces[edge.func_] << label_delim
					<< nodes_[edge.func_].label_ << ','
					<< nodeidces[edge.child_] << label_delim
					<< nodes_[edge.child_].label_ << ','
					<< edge.child_idx_ << ','
					<< color << '\n';
			}
			else
			{
				out << nodeidces[edge.func_] << label_delim
					<< nodes_[edge.func_].label_ << ','
					<< nnodes + i << label_delim
					<< coorders_[edge.coorder_] << ','
					<< edge.child_idx_ << ','
					<< color << '\n';

				out << nnodes + i << label_delim
					<< coorders_[edge.coorder_] << ','
					<< nodeidces[edge.child_] << label_delim
					<< nodes_[edge.child_].label_ << ','
					<< edge.child_idx_ << ','
					<< color << '\n';
			}
		}
	}

	bool showshape_ = false;

private:
	struct Edge
	{
		Edge (ade::iFunctor* func,
			ade::iTensor* child,
			ade::iCoordMap* coorder,
			size_t child_idx) :
			func_(func),
			child_(child),
			coorder_(coorder),
			child_idx_(child_idx) {}

		bool operator == (const Edge& other) const
		{
			return func_ == other.func_
				&& child_ == other.child_
				&& coorder_ == other.coorder_
				&& child_idx_ == other.child_idx_;
		}

		ade::iFunctor* func_;

		ade::iTensor* child_;

		ade::iCoordMap* coorder_;

		size_t child_idx_;
	};

	struct Node
	{
		std::string label_;

		NODE_TYPE ntype_;
	};

	struct EdgeHash
	{
		size_t operator() (const Edge& edge) const
		{
			std::stringstream ss;
			ss << std::hex
				<< edge.func_
				<< edge.child_
				<< edge.coorder_
				<< edge.child_idx_;
			return std::hash<std::string>()(ss.str());
		}
	};

	std::unordered_map<ade::iTensor*,Node> nodes_;

	std::unordered_map<ade::iCoordMap*,std::string> coorders_;

	std::unordered_set<Edge,EdgeHash> edges_;

	GetTypeFuncT get_ftype_;
};
