/*********************************************************************************
 * File: Graph.hpp
 * Description: Data structure for partial graph (candidate set) of an image
 * by BreakDS, University of Wisconsin Madison, Sat Aug 18 18:41:25 CDT 2012
 *********************************************************************************/

#pragma once
#include <string>
#include <vector>

namespace PatTk
{
  struct PatLoc
  {
    int index;
    float y, x, scale, rotation;
    PatLoc( int index_, float y_, float x_, float scale_, float rotation_ )
      : index(index_), y(y_), x(x_), scale(scale_), rotation(rotation_) {}
    PatLoc() : index(0), y(0), x(0), scale(0), rotation(0) {}
  };

  
  class PatGraph
  {
  private:
    std::vector<std::vector<PatLoc> > pool;
  public:
    int rows, cols;

  public:

    // Constructor
    PatGraph( int h, int w ) : rows(h), cols(w)
    {
      pool.resize( h * w );
      for ( int i=0; i<h*w; i++ ) pool[i].clear();
    }

    PatGraph( std::string &filename )
    {
      WITH_OPEN( in, filename, "r" );
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
        }
      }
      END_WITH( in );
    }

  public:
    // I/O operations
    void write( std::string &filename )
    {
      WITH_OPEN( out, filename, "w" );
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
        }
      }
      END_WITH( out );
    }
    
  };
};

