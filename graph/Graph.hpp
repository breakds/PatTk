/*********************************************************************************
 * File: Graph.hpp
 * Description: Data structure for partial graph (candidate set) of an image
 * by BreakDS, University of Wisconsin Madison, Sat Aug 18 18:41:25 CDT 2012
 *********************************************************************************/

#pragma once
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "../interfaces/cv_interface.hpp"

using namespace EnvironmentVariable;

namespace PatTk
{

  struct AffineTransform
  {
    float a, b, dy, dx;

    /*
      | a  b dy |
      | -b a dx |
      | 0  0 1  |
    */
    AffineTransform() : a(1.0f), b(0.0f), dy(0.0f), dx(0.0f) {}

    AffineTransform( float a_, float b_, float dy_, float dx_ ) :
      a(a_), b(b_), dy(dy_), dx(dx_) {}

    
    inline void write( FILE* out )
    {
      fwrite( &a, sizeof(float), 1, out );
      fwrite( &b, sizeof(float), 1, out );
      fwrite( &dy, sizeof(float), 1, out );
      fwrite( &dx, sizeof(float), 1, out );
    }

    inline void read( FILE* in )
    {
      fread( &a, sizeof(float), 1, in );
      fread( &b, sizeof(float), 1, in );
      fread( &dy, sizeof(float), 1, in );
      fread( &dx, sizeof(float), 1, in );
    }

    inline void apply( float y, float x, float& new_y, float& new_x ) const
    {
      new_y = a * y + b * x + dy;
      new_x = -b * y + a * x + dx;
    }
  };
  
  struct PatLoc
  {
    int index;
    int y;
    int x;
    float scale, rotation, dist;

    PatLoc( int index_, int y_, int x_, float scale_, float rotation_, float dist_ )
      : index(index_), y(y_), x(x_), scale(scale_), rotation(rotation_), dist(dist_) {}

    inline PatLoc( FILE *in )
    {
      read( in );
    }

    PatLoc() : index(0), y(0), x(0), scale(0), rotation(0), dist(0) {}

    // constructor with shift
    PatLoc( const PatLoc& loc, int dy, int dx )
      : index(loc.index), scale(loc.scale), rotation(loc.rotation)
    {
      double S = 1.0 / scale;
      double cosa = cos( -rotation ) * S;
      double sina = sin( -rotation ) * S;
      y = loc.y + dy * cosa + dx * sina;
      x = loc.x - dy * sina + dx * cosa;
    }

    inline void GetTransform( float *transform, int i, int j ) const
    {
      static int radius = env["patch-w"] >> 1;
      double S = 1.0 / scale;

      double cosa = cos( -rotation ) * S;
      double sina = sin( -rotation ) * S;

      // the top-left corner of the target patch
      double y1 = radius * ( - cosa - sina + 1 ) + y;
      double x1 = radius * ( sina - cosa + 1 ) + x;

      /*
       * | S*cosa   S*sina   dy  |    | i |    | y1 |
       * | -S*sina  S*cosa   dx  | x  | j | =  | x1 |
       * |   0        0      1   |    | 1 |    |  1 |
       */
      
      transform[0] = static_cast<float>( index ); // index
      transform[1] = sina * S;
      transform[2] = cosa * S;
      transform[3] = scale;
      transform[4] = y1 - i * S * cosa - j * S * sina; // dy
      transform[5] = x1 + i * S * sina - j * S * cosa; // dx
    }


    inline Image<BGRCell,int,false>::Patch toPatch( const Album<BGRCell, int, false> &album ) const
    {

      // cv::Mat raw = cv::imread( strf( "%s/%s_L.png", env["dataset"].c_str(),
      //                                 imgList[index].c_str() ) );
      // if ( raw.empty() ) {
      //   Error( "cannot open image %s/%s.png", env["dataset"].c_str(),
      //          imgList[index].c_str() );
      //   exit( -1 );
      // }
      
      //      Image<BGRCell, int, false > img = std::move( cvFeatGen<BGRCell, int, false>::gen( raw ) );

      static int radius = env["patch-w"] >> 1;
      
  
      double ang = - rotation;
      double cosa = cos( ang ) * scale;
      double sina = sin( ang ) * scale;
  
      double y1 = ( -radius * cosa - radius * sina + ( y + radius ) );
      double x1 = ( radius * sina - radius * cosa + ( x + radius ) );

      return album(index).Spawn( y1, x1, 1.0 / scale, ang / M_PI * 180.0 );
    }

    

    inline void write( FILE *out ) const
    {
      static const float inv_pi_mul_90 = static_cast<float>( 90.0 / M_PI );
      fwrite( &index, sizeof(int), 1, out );

      // only the first 16 bits of y and x are written out
      fwrite( &y, 2, 1, out );
      fwrite( &x, 2, 1, out );

      fwrite( &scale, sizeof(float), 1, out );

      // The rotation is rounded to the nearest multiple of 2 degrees
      unsigned char cTmp = static_cast<unsigned char>( rotation * inv_pi_mul_90 + 0.5 );
      fwrite( &cTmp, sizeof(unsigned char), 1, out );

      fwrite( &dist, sizeof(float), 1, out );
    }

    inline void read( FILE *in )
    {
      static const float pi_div_90 = static_cast<float>( M_PI / 90.0 );
      fread( &index, sizeof(int), 1, in );

      // read only 16 bits for both y and x
      fread( &y, 2, 1, in );
      fread( &x, 2, 1, in );

      fread( &scale, sizeof(float), 1, in );

      // the rotattion is rounded to the nearest multiple of 2 degrees
      unsigned char cTmp = 0;
      fread( &cTmp, sizeof(unsigned char), 1, in );
      rotation = cTmp * pi_div_90;

      fread( &dist, sizeof(float), 1, in );
    }

    inline bool operator==( const PatLoc &other ) const
    {
      if ( index != other.index ) return false;
      if ( scale - other.scale > 1e-5 ) return false;
      if ( rotation - other.rotation > 1e-5 ) return false;
      if ( y - other.y > 1e-5 ) return false;
      if ( x - other.x > 1e-5 ) return false;
      if ( dist - other.dist > 1e-5 ) return false;
      return true;
    }

    // For debugging:
    void show() const
    {
      printf( "[id=%4d], y:%d, x:%d, scale: %.2f, rotation: %.2f, dist: %.4f\n",
              index, y, x, scale, rotation, dist );
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
    
    PatGraph( const std::string &filename )
    {
      WITH_OPEN( in, filename.c_str(), "r" );
      fread( &rows, sizeof(int), 1, in );
      fread( &cols, sizeof(int), 1, in );
      pool.resize( rows * cols );
      for ( auto& ele : pool ) {
        unsigned char len = 0;
        fread( &len, sizeof(unsigned char), 1, in );
        ele.resize( len );
        for ( auto& cand : ele ) cand.read( in );
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
      cols = other.cols;
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
        unsigned char len = static_cast<unsigned char>( ele.size() );
        fwrite( &len, sizeof(unsigned char), 1, out );
        for ( auto& cand : ele ) cand.write( out );
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

