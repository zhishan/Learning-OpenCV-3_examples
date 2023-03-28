#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static inline int gcd(int a, int b)                                                                                                                                                                        
{
    if( a < b )
        std::swap(a, b);
    while( b > 0 )
    {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

/** @brief Aligns a buffer size to the specified number of bytes.

The function returns the minimum number that is greater than or equal to sz and is divisible by n :
\f[\texttt{(sz + n-1) & -n}\f]
@param sz Buffer size to align.
@param n Alignment size that must be a power of two.
 */
static inline size_t alignSize1(size_t sz, int n)
{
    cout << "n=" << n << endl;
    assert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n-1) & -n;
}


int main() {
    Size winStride(20, 20); //winSize
    Size blockStride(8, 8); //blocksize
     
    Size cacheStride(gcd(winStride.width, blockStride.width),
                                 gcd(winStride.height, blockStride.height)); 

  cout << "cacheStride=" << cacheStride << endl;
  cout << "alignSize width = " << alignSize1(0, cacheStride.width) << endl;
  cout << "alignSize height = " << alignSize1(0, cacheStride.height) << endl;
}
