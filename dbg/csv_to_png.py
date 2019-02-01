''' Conversion script for writing ade_csv format to png '''

import sys
from collections import defaultdict

import graphviz as gv

_styles = {
    'graph': {
        'label': 'Operation Graph',
        'fontsize': '16',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'rankdir': 'BT',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'hexagon',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    },
    'edges': {
        'style': 'dashed',
        'color': 'white',
        'arrowhead': 'open',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'white',
    }
}

def _str_clean(str):
    # get rid of <, >, |, :
    str = str.strip()\
        .replace('<', '(')\
        .replace('>', ')')\
        .replace('|', '!')\
        .replace('\\', ',')\
        .replace(':', '=')\
        .replace(',,', '\\n')
    return str

def _apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph

def read_graph(lines):
    nodes = set()
    edges = defaultdict(list)
    for line in lines:
        cols = line.split(',')
        if len(cols) != 3: # ignore ill-formatted lines
            continue # todo: warn
        observer, subject, order = tuple(col for col in cols)
        obs = _str_clean(observer)
        sub = _str_clean(subject)
        nodes.add(obs)
        nodes.add(sub)
        edges[obs].append((sub, order))
    return (nodes, edges)

def print_graph(callgraph, outname):
    nodes, edges = callgraph

    g1 = gv.Digraph(format='png')
    for node in nodes:
        g1.node(node)

    for observer in edges:
        for subject, idx in edges[observer]:
            g1.edge(observer, subject, idx)

    _apply_styles(g1, _styles)
    g1.render(outname, view=True)

def csv_to_png(lines, outpath):
    print_graph(read_graph(lines), outpath)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', nargs='?', default=None,
        help='Path of the CSV-formated edgegraph to parse (Default: None)')
    parser.add_argument('--out', nargs='?', default='opgraph',
        help='Path of the output file excluding extension (Default: opgraph)')
    args = parser.parse_args()

    with (open(args.csv) if args.csv else sys.stdin) as infile:
        csv_to_png(infile, args.out)
