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
  
  ```C++
  for (int i = 0; i < 1000; i++) {
    cml::vec3f a = A[i];
    cml::vec3f b = B[i];        
    
    cml::vec3f v = cml::normalize(cml::cross(a, b));
    
    C[i] = v;
  }
  ```

The original intent of this library was to make my own life easier when using CUDA. It's also been a lot of fun messing around with C++ templates. I hope others who happen to stumble across this will also find it as useful.

A lot of what you see here was inspired/taken from tools that already exist, such as the popular [glm library](https://github.com/g-truc/glm) and from the fantastic book [Physically Based Rendering: From Theory To Implementation](https://www.pbr-book.org/).

Please feel free to open an issue or pull request if you have any suggestions or find any bugs.

### Current road map
- Better unit test coverage
  -  ~Vectors~
  -  ~Complex Numbers~
  -  Qauternions
  -  2D Matrix
  -  3D Matrix
  -  4D Matrix
- Speed testing
- Graphics specific functions
- ~2D and 3D Matrix implementation~
