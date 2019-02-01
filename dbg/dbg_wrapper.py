from csv_to_png import csv_to_png
from dbg.dbg import graph_to_csvstr

def graph_to_csvimg(root, outpath, showshape = False):
    lines = graph_to_csvstr(root, showshape).split('\n')
    csv_to_png(lines, outpath)
