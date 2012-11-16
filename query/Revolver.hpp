/*********************************************************************************
 * File: Revolver.hpp
 * Description: 1) A random number pool maintainer, maintans groups of random number pool  
 *              2) Capable of group of sample generator without replacement
 * by BreakDS, @ University of Wisconsin-Madison, Apr 6 2012
 * ======================================================================
 *********************************************************************************/


#pragma once

#include <cstdlib>
#include <vector>

using std::vector;

#define SHUFFLER_ERROR UINT_MAX

class Shuffler
{
private:
  uint size;
  vector<uint> pool;
  uint curPos;
  // pool[address] = id
public:

  inline Shuffler()
  {
    size = 0;
    pool.clear();
  }
  
  inline Shuffler( const vector<uint> &init ) 
  {
    pool.clear();
    size = init.size();
    for ( auto& ele : init ) {
      pool.push_back( ele );
    }
  }

  inline void Reset( const vector<uint> &init ) 
  {
    pool.clear();
    size = init.size();
    for ( auto& ele : init ) {
      pool.push_back( ele );
    }
  }

  
  inline void Reset( uint n )
  {
    size = n;
    pool.resize( n );
    for ( uint i=0; i<n; i++ ) {
      pool[i] = i;
    }
  }
  
  inline Shuffler( uint n ) 
  {
    size = n;
    pool.resize( n );
    for ( uint i=0; i<n; i++ ) {
      pool[i] = i;
    }
  }

  inline Shuffler( const Shuffler &sfl ) 
  {
    size = sfl.size;
    pool.resize( sfl.size );
    for ( uint i=0; i<sfl.size; i++ ) {
      pool[i] = sfl(i);
    }
  }

  inline void operator=( const Shuffler &sfl )
  {
    size = sfl.size;
    pool.resize( sfl.size );
    for ( uint i=0; i<sfl.size; i++ ) {
      pool[i] = sfl(i);
    }
  }
  
  inline uint Number() const
  {
    return size;
  }
  
  // splice by specifying the id
  // involves looking up
  inline void SpliceID( uint id )
  {
    for ( uint i=0; i<size-1; i++ ) {
      if ( id == pool[i] ) {
        pool[i] = pool[size-1];
        pool[size-1] = id;
        break;
      }
    }
    size--;
  }

  // splice by specifying the address
  // no looking up involved
  inline void SpliceAddress( uint add )
  {
    uint t = pool[add];
    pool[add] = pool[size-1];
    pool[size-1] = t;
    size--;
  }
  
  inline void Shuffle( uint k ) 
  {
    for ( uint i=0, end= k < size ? k : size; i < end; i++  ) {
      uint r = rand() % (size-i) + i;
      uint t = pool[r];
      pool[r] = pool[i];
      pool[i] = t;
    }
  }

  inline void Shuffle() 
  {
    for ( uint i=0; i<size; i++  ) {
      uint r = rand() % (size-i) + i;
      uint t = pool[r];
      pool[r] = pool[i];
      pool[i] = t;
    }
  }

  inline void Keep( uint k ) {
    if ( k < size ) {
      size = k;
    }
  }
  
  // Fisher-Yates shuffle
  // - Start Shuffle
  inline void ResetShuffle() {
    curPos = -1;
  }
  
  // - Get Next sample
  inline uint Next()
  {
    curPos++;
    if ( curPos >= size ) return SHUFFLER_ERROR; // Out of bound
    uint r = rand() % (size-curPos) + curPos;
    uint t = pool[r];
    pool[r] = pool[curPos];
    pool[curPos] = t;
    return t;
  }

  // - Delete sample at current position
  inline void Disqualify()
  {
    SpliceAddress( curPos );
  }




  inline uint operator() ( uint add ) const
  {
    return pool[add];
  }
  
  
};




