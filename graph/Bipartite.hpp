/*********************************************************************************
 * File: Bipartite.hpp
 * Description: Create a undirected bi-partition graph with weighted edges
 * by BreakDS, University of Wisconsin Madison, Tue Dec 11 16:12:43 CST 2012
 *********************************************************************************/

#pragma once

#include <vector>
#include <utility>

using std::vector;
using std::pair;
using std::make_pair;

namespace PatTk
{
  class Bipartite
  {
  private:
    vector< vector< pair< int, double > > > a_to_b;
    vector< vector< pair< int, double > > > b_to_a;

  public:
    Bipartite( int numA, int numB )
    {
      a_to_b.resize( numA );
      for ( int i=0; i<numA; i++ ) {
        a_to_b[i].clear();
      }

      b_to_a.resize( numB );
      for ( int i=0; i<numB; i++ ) {
        b_to_a[i].clear();
      }
    }

    inline void grow_a( int target ) {
      int old = static_cast<int>( a_to_b.size() );
      a_to_b.reserve( target );
      for ( int i=old; i<target; i++ ) {
        a_to_b.push_back( vector< pair<int, double > >() );
        a_to_b.back().clear();
      }
    }

    inline void grow_b( int target ) {
      int old = static_cast<int>( b_to_a.size() );
      b_to_a.reserve( target );
      for ( int i=old; i<target; i++ ) {
        b_to_a.push_back( vector< pair<int, double > >() );
        b_to_a.back().clear();
      }
    }

    
    inline void add( int a, int b, double wt )
    {
      if ( a >= static_cast<int>( a_to_b.size() ) ) {
        grow_a( a + 1 );
      }
      if ( b >= static_cast<int>( b_to_a.size() ) ) {
        grow_b( b + 1 );
      }
      a_to_b[a].push_back( make_pair( b, wt ) );
      b_to_a[b].push_back( make_pair( a, wt ) );
    }

    inline const vector<pair<int,double> >& getToSet( int a ) const
    {
      return a_to_b[a];
    }

    inline const vector<pair<int,double> >& getFromSet( int b ) const
    {
      return b_to_a[b];
    }
  };
}
