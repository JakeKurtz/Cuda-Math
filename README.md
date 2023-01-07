# C++/CUDA template math library

This is a humble little math library that is designed to provide efficient and flexible code while having an easy-to-use interface for scientific and graphic applications. 

So far, the library includes interfaces and functions for working with following types

- 2d, 3d, 4d Vectors
- 4d Matrices
- Random number generation
- Sampling
- Quaternions
- Complex numbers

I've also integrated the [ASA container developed by Robert Strzodkaf](https://asc.ziti.uni-heidelberg.de/node/18). It allows users to easily switch from AoS to SoA with a single line of code e.g.

  ```C++
  typedef ASX::Array<ASX::SOA, 1000, cml::vec3f> vec3_array; // Structure of Arrays Memory Layout.
  typedef ASX::Array<ASX::AOS, 1000, cml::vec3f> vec3_array; // Array of Structures Memory Layout.
  ```

And regardless of the layout specified, objects always act as if they were structs
  ```C++
  for (int i = 0; i < 1000; i++) {
    C[i] += cml::dot(A[i], B[i]);
  }
  ```

The intent of this library is to make my own life easier while playing around with CUDA and OpenGL. I hope others who stumble across this will also find it useful.

A lot of what you see here was inspired/taken from tools that already exist, such as the popular [glm library](https://github.com/g-truc/glm) and from the fantastic book [Physically Based Rendering: From Theory To Implementation by Matt Pharr, Wenzel Jakob, and Greg Humphreys](https://www.pbr-book.org/).

Please feel free to open an issue or pull request if you have any suggestions or find any bugs.

### Current road map
- Better unit test coverage
- Speed testing
- 2D and 3D Matrix implementation
