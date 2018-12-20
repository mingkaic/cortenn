# PBM (Protobuf Marshaller)

Serialize ADE graphs created by top-level code in protobuf format.

Saving and loading requires data serialization functionals as parameters. This parameterization is to defer data formatting responsibilities to the library implementing ADE.

## Why Protobuf

Because protobuf parsers is consistent across all popular languages.

## Extension

User libraries need to provide an encoding and decoding functions for the library's generic data format when saving and loading
