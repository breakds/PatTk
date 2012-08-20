/*********************************************************************************
 * File: Graph.hpp
 * Description: Data structure for partial graph (candidate set) of an image
 * by BreakDS, University of Wisconsin Madison, Sat Aug 18 18:41:25 CDT 2012
 *********************************************************************************/

#pragma once
#include <string>
#include <vector>
#include "LLPack/utils/extio.hpp"

namespace PatTk
{
  struct PatLoc
  {
    int index;
    float y, x, scale, rotation, dist;
    PatLoc( int index_, float y_, float x_, float scale_, float rotation_, float dist_ )
      : index(index_), y(y_), x(x_), scale(scale_), rotation(rotation_), dist(dist_) {}
    PatLoc() : index(0), y(0), x(0), scale(0), rotation(0), dist(0) {}
    void show() const
    {
      printf( "y:%.2f, x:%.2f, scale: %.2f, rotation: %.2f, dist: %.4f\n", y, x, scale, rotation, dist );
    }
  };
  
  
  class PatGraph
  {
  private:
    std::vector<std::vector<PatLoc> > pool;
    std::vector<int> pos;
  public:
    int rows, cols;

  private:
    // prohibit calling copy construtor and copy assignment
    PatGraph( const PatGraph& other );
    const PatGraph& operator=( const PatGraph& other );

  public:

    // Constructor
    PatGraph( int h, int w ) : rows(h), cols(w)
    {
      pool.resize( h * w );
      for ( int i=0; i<h*w; i++ ) pool[i].clear();
      pos.resize(h);
      for ( int i=0; i<h; i++ ) pos[i] = i * w;
    }

    PatGraph( std::string &filename )
    {
      WITH_OPEN( in, filename.c_str(), "r" );
      fread( &rows, sizeof(int), 1, in );
      fread( &cols, sizeof(int), 1, in );
      pool.resize( rows * cols );
      for ( auto& ele : pool ) {
        int len = 0;
        fread( &len, sizeof(int), 1, in );
        ele.resize( len );
        for ( auto& cand : ele ) {
          fread( &cand.index, sizeof(int), 1, in );
          fread( &cand.x, sizeof(float), 1, in );
          fread( &cand.y, sizeof(float), 1, in );
          fread( &cand.scale, sizeof(float), 1, in );
          fread( &cand.rotation, sizeof(float), 1, in );
          fread( &cand.dist, sizeof(float), 1, in );
        }
      }
      pos.resize( rows );
      for ( int i=0; i<rows; i++ ) pos[i] = i * cols;
      END_WITH( in );
    }
    
    // Move Constructor
    PatGraph( PatGraph&& other ) noexcept
    {
      rows = other.rows;
      cols = other.cols;
      pool.swap( other.pool );
      pos.swap( other.pos );
    }

    // Move Assignment
    const PatGraph& operator=( PatGraph&& other )
    {
      rows = other.rows;
      cols = other.rows;
      pool.swap( other.pool );
      pos.swap( other.pos );
      return (*this);
    }
    
  public:
    // I/O operations
    void write( std::string &filename )
    {
      WITH_OPEN( out, filename.c_str(), "w" );
      fwrite( &rows, sizeof(int), 1, out );
      fwrite( &cols, sizeof(int), 1, out );
      for ( auto& ele : pool ) {
        int len = static_cast<int>( ele.size() );
        fwrite( &len, sizeof(int), 1, out );
        for ( int i=0; i<len; i++ ) {
          fwrite( &ele[i].index, sizeof(int), 1, out );
          fwrite( &ele[i].x, sizeof(float), 1, out );
          fwrite( &ele[i].y, sizeof(float), 1, out );
          fwrite( &ele[i].scale, sizeof(float), 1, out );
          fwrite( &ele[i].rotation, sizeof(float), 1, out );
          fwrite( &ele[i].dist, sizeof(float), 1, out );
        }
      }
      END_WITH( out );
    }

  public:
    // Selectors and Operators
    const std::vector<PatLoc>& operator()( int y, int x ) const
    {
      return pool[pos[y]+x];
    }
    
    const std::vector<PatLoc>& operator()( int i ) const
    {
      return pool[i];
    }
    
    std::vector<PatLoc>& operator[]( int i )
    {
      return pool[i];
    }


    // Merge Operator
    const PatGraph& operator+=( const PatGraph& other )
    {
      assert( rows == other.rows && cols == other.cols );
      for ( int i=0; i<rows*cols; i++ ) {
        for ( auto& ele : other(i) ) {
          pool[i].push_back( ele );
        }
      }
      return (*this);
    }
  };
};

