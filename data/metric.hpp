
#pragma once
#include <cmath>
#include "LLPack/utils/candy.hpp"
#include "2d.hpp"

namespace PatTk
{

  template <typename cellType, typename valueType, bool lite = true>
  inline double FakeLabelDist( typename Image<cellType,valueType,lite>::Patch &a,
                               typename Image<cellType,valueType,lite>::Patch &b,
                               typename std::enable_if<lite>::type __attribute__((__unused__)) *padding=0 )
  {
    double l2 = sqrt( ( a.y - b.y ) * ( a.y - b.y ) + ( a.x - b.x ) * ( a.x - b.x ) );
    if ( a.pid != b.pid ) {
      return 10000000.0 + l2;
    }
    return l2;
  }
  
};



// template <typename cellType>
// class CellDist
// {
//   static_assert( std::is_base_of< AbstractCell<typename cellType::type>, cellType >::value,
//                  "cellType is not a valid cell type. (does not derive from AbstractCell." );
// public:
//   typedef typename cellType::type data_t;
//   typedef typename cellType cell_t;
//   virtual typename std::enable_if<std::is_integral<typename cellType::type>::value, int>
//   operator()( const cellType& a, const cellType& b ) = 0;
//   virtual typename std::enable_if<std::is_floating_point<typename cellType::type>::value, double>
//   operator()( const cellType& a, const cellType& b ) = 0;
// };



// template <typename cellType>
// class L2 : public CellDist<cellType>
// {
// public:
//   typename std::enable_if<std::is_integral<typename cellType::type>::value, int>
//   operator()( const cellType& a, const cellType& b )
//   {
//     assert( a.length == b.length );
//     int accu(0), tmp(0);
//     for ( int i=0; i<a.length; i++ ) {
//       tmp = static_cast<int>(a[i]) - static_cast<int>(b[i]);
//       accu += tmp * tmp;
//     }
//     return accu;
//   }

//   typename std::enable_if<std::is_integral<typename cellType::type>::value, double>
//   operator()( const cellType& a, const cellType& b )
//   {
//     assert( a.length == b.length );
//     double accu(0), tmp(0);
//     for ( int i=0; i<a.length; i++ ) {
//       double = static_cast<double>(a[i]) - static_cast<double>(b[i]);
//       accu += tmp * tmp;
//     }
//     return accu;
//   }
// };

// template <typename cellDistType>
// class PatchDist
// {
// public:
//   operator()
// }


  

