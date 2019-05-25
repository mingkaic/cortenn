# Cortenn
[![Build Status](https://travis-ci.org/mingkaic/cortenn.svg?branch=master)](https://travis-ci.org/mingkaic/cortenn)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/cortenn/badge.svg?branch=master)](https://coveralls.io/github/mingkaic/cortenn?branch=master)

## Synopsis

Cortenn implements data formatting and manipulation portions for the [Tenncor libraries](https://github.com/mingkaic/tenncor) help developers write math equations for machine learning.

This project is a multiplex layer for Tenncor and some data-manipulation library. This library exist to avoid changing Tenncor too much.

## Components

- [AGE (ADE Generation Engine)](age/README_AGE.md)

This generator creates glue layer between ADE and data manipulation libraries as well as map operational codes to its respective chain rule.

- [BWD (Backward Operations)](bwd/README_BWD.md)

This library provides traveler for generating partial derivative equations using some set of chain rules.

- [LLO (Low Level Operators)](llo/README_LLO.md)

This module is implements basic operations for Tenncor's ADE Tensor objects generated through pybinder.
Additionally, llo also defines data format and (de)serialization methods required by PBM.

- [Pybinder](pybinder/README_PY.md)

This generator extends Tenncor's AGE generator. In this instance, on top of generating the ADE operators specified in LLO, pybinder generates pybind11 binding code.

## Building

Cortenn uses bazel 0.15+.

Download bazel: https://docs.bazel.build/versions/master/install.html
