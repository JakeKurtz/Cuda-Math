# C++/CUDA template math library

This is a humble little math library that is designed to provide efficient and flexible code while having an easy-to-use interface for scientific and graphic applications.

So far, the library includes interfaces and functions for working with following types

- 2d, 3d, 4d Vectors
- 2d, 3d, 4d Matrices
- Random number generation
- Sampling
- Quaternions
- Complex numbers

The main purpose of this project, is to be used as a tool to teach myself better C++ and CUDA, while at the same time, gaining a better understanding on the core mathematics for graphic/scientific applications. A lot of what you see here was inspired/taken from tools that already exist, such as the popular [glm library](https://github.com/g-truc/glm) and from the fantastic book [Physically Based Rendering: From Theory To Implementation by Matt Pharr, Wenzel Jakob, and Greg Humphreys](https://www.pbr-book.org/).

Please feel free to open an issue or pull request if you have any suggestions or find any bugs.

### Current road map
- Better unit test coverage
- 2D and 3D Matrix implementation
- [Custom containers for each type, that can easily switch from SoA to AoS](https://asc.ziti.uni-heidelberg.de/node/18).
