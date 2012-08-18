/*********************************************************************************
 * File: DT.hpp
 * Description: Distance Transform
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 17 08:48:13 CDT 2012
 *********************************************************************************/



#pragma once
#include <memory>
#include <type_traits>
#include "LLPack/algorithms/sort.hpp"

using std::vector;

namespace optimize
{

  template <typename floating=float>
  class AbstractDT
  {
  public:
    virtual void operator()( const vector<floating> &h, const vector<floating> &a, const vector<floating> &b, int K,
                             vector<int> &match ) = 0;
  };

  /*
   * Solving min_{i=1..K} h_i + ||a_i-b_j||, for j=1..K
   * Input:  h[1..K], a[1..K], b[1..K]
   * Output: match[1..K], where match[j] = arg min_i
   */
  template<typename floating=float>
  class FDT : public AbstractDT<floating>
  {
  public:
    void operator()( const vector<floating> &h, const vector<floating> &a, const vector<floating> &b, int K,
              vector<int> &match )
    {

      // Sorting:
      vector<int> inda = std::move( sorting::index_sort( a, true ) );

      // Lower envlope:
      vector<int> cones;
      cones.reserve(K);
      vector<floating> knots;
      knots.reserve(K);
      for ( int i=0; i<K; i++ ) {
        // Pop stack if the stack top cone doesn't intersect with the incoming one
        // Whether they'll intersect is determined by comparing horizontal displacement
        // and vertical displacement of the two apexes.
        floating horz(0), vert(0);
        while ( ! cones.empty() ) {
          horz = a[inda[i]] - a[cones.back()];
          vert = h[cones.back()] - a[inda[i]];
          if ( vert < horz ) {
            break;
          } else {
            cones.pop_back();
            knots.pop_back();
          }
        }

        cones.push_back( inda[i] );
        if ( cones.empty() ) {
          knots.push_back( 0 );
        } else {
          // Find the intersection point
          // let x = horizontal displacement from left apex to the intersection point
          //     y = horizontal displacement from right apex to the intersection point
          // Then,
          // x + y = horz
          // x + vert = y
          // => x = (horz-vert) * 0.5, y = (horz+vert) * 0.5
          // => intersect = a[cones.back()] + x;
          knots.push_back( a[cones.back()] + (horz-vert) * 0.5 );
        }
      }

      // fill "match" array
      vector<int> indb = std::move( sorting::index_sort( b, true ) );
      int num = static_cast<int>( knots.size() );
      int i = 1;
      for ( int j=0; j<K; j++ ) {
        while ( i < num && knots[i] < b[indb[j]] ) {
          i++;
        }
        match[j] = cones[i-1];
      }
    };
  };
};
   
