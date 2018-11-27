# PBM (Protobuf Marshaller)

Serialize ADE graphs created by top-level code in protobuf format.

Protobuf is platform independent format for structured data.

## Why Protobuf

Because the library can be used across all the languages I'm familiar with. It's also more transparent and less bug-ridden than most free JSON parsing libraries. Also JSON face potential precision-loss for floating point data

## Extension

User libraries need to provide an encoding and decoding functions for the library's generic data format when saving and loading
