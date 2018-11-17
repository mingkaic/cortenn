# Cortenn

## Synopsis

Cortenn extends the Tenncor libraries help developers write math equations for machine learning

## Components

- [LLO (Low Level Operators)](llo/README_LLO.md)

This module is a sample library of data operators mapped to the ADE opcodes.
Expect this module to split when I decide to depend on external libraries (like eigen).

- [PBM (Protobuf Marshaller)](pbm/README_PBM.md)

This module marshals llo-extended graph

## Building

Cortenn uses bazel 0.15+.

Download bazel: https://docs.bazel.build/versions/master/install.html
