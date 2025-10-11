# Contributing to PageANN

Thank you for your interest in contributing to PageANN! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you encounter bugs or have feature requests:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Detailed description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, compiler version, etc.)
   - Relevant code snippets or error messages

### Submitting Code Changes

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**:
   - Follow the existing code style and conventions
   - Add comments for complex logic
   - Update documentation as needed
   - Add tests if applicable

3. **Test your changes**:
   - Ensure all existing tests pass
   - Add new tests for new features
   - Test on relevant platforms (Linux/Windows)

4. **Commit your changes**:
   - Write clear, concise commit messages
   - Reference issue numbers in commits (e.g., "Fix #123: ...")
   - Keep commits focused and atomic

5. **Submit a Pull Request**:
   - Provide a clear description of changes
   - Reference related issues
   - Explain the rationale for your changes
   - Include any breaking changes or migration notes

## Code Style Guidelines

### C++ Code
- Follow existing indentation (4 spaces, no tabs)
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Add header comments for new files:
  ```cpp
  // Copyright (c) Microsoft Corporation. All rights reserved.
  // Licensed under the MIT license.
  //
  // PageANN: Page-based Index Search Engine
  // Copyright (c) 2025 Dingyi Kang <dingyikangosu@gmail.com>. All rights reserved.
  // Licensed under the MIT license.
  ```

### Documentation
- Update README.md for significant features
- Update workflow documentation in `workflows/` folder
- Comment complex algorithms and data structures
- Keep documentation clear and concise

## Development Setup

### Prerequisites
- CMake 3.15+
- C++17 compatible compiler (GCC 9+, Clang 9+, MSVC 2019+)
- Boost libraries
- Intel MKL (for BLAS operations)
- libaio (Linux only)

### Building from Source
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Testing

Before submitting a PR, ensure:
- Code compiles without warnings
- Existing functionality is not broken
- New features include appropriate tests
- Performance-critical changes include benchmarks

## Areas for Contribution

We welcome contributions in:

- **Performance improvements**: Optimization of search algorithms, I/O operations
- **New features**: Additional distance metrics, index formats, search strategies
- **Documentation**: Tutorials, examples, API documentation
- **Bug fixes**: Addressing reported issues
- **Platform support**: Windows compatibility, new architecture support
- **Testing**: Unit tests, integration tests, benchmark suites

## Questions and Discussions

- **Email**: dingyikangosu@gmail.com
- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code discussions

## License

By contributing to PageANN, you agree that your contributions will be licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

PageANN builds upon Microsoft's DiskANN. We are grateful for their open-source contributions to the vector search community.
