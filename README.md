# JSON Search Engine

A high-performance C++ search engine designed for structured JSON datasets with intelligent matching capabilities and extensible architecture.

## Features

### Core Functionality
- **Dynamic JSON Parsing**: Load and process any JSON structure with automatic key-value mapping
- **Dual Search Modes**: 
  - **Exact Match**: Precise field-based searches with 100% accuracy
  - **Fuzzy Search**: Intelligent partial keyword matching with relevance scoring
- **Case-Insensitive Processing**: Robust search handling with automatic normalization
- **Memory Efficient**: Optimized data structures for large datasets
- **Cross-Platform**: Compatible with Windows, macOS, and Linux

### Advanced Capabilities
- **Multi-Field Search**: Search across any number of JSON fields simultaneously
- **Type Flexibility**: Handle strings, numbers, and nested objects seamlessly
- **Error Resilience**: Graceful handling of malformed JSON with detailed error reporting
- **Performance Optimized**: O(n) search complexity with potential for indexing upgrades

##  Technical Stack

- **Language**: C++17/20
- **JSON Library**: nlohmann/json (header-only)
- **Compiler Support**: GCC 7+, Clang 6+, MSVC 2019+

