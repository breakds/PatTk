/*********************************************************************************
 * File: Trans.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Relative transformation between images.
 *              Data structures and operations.
 *********************************************************************************/

#pragma once
#include <vector>
#include "../query/tree.hpp"


namespace PatTk
{


  class PatLoc
  {
  public:
    int id, y, x;
    float rotation, scale;
    PatLoc( int id1, int y1, int x1, float rot1, float sc1 )
      : id(id1), y(y1), x(x1), rotation(rot1), scale(sc1) {}

    PatLoc( const LocInfo& loc ) :
      id(loc.id), y(loc.y), x(loc.x), rotation(0.0f), scale(1.0f) {}

    PatLoc() : id(-1), y(0), x(0), rotation(0.0f), scale(1.0f) {}
    
    void show() const
    {
      printf( "id: %d, (%d,%d), rotation: %.2f, scale: %.2f\n",
              id, y, x, rotation / M_PI * 180.0, scale );
    }

  };


  class Sampler4D
  {
  private:
    int rows, cols;
    float y_wnd, x_wnd, rot_wnd, scale_wnd;

  public:
    inline Sampler4D( int h, int w )
      : rows(h), cols(w),
        y_wnd(static_cast<float>(h>>1)),
        x_wnd(static_cast<float>(w>>1)),
        rot_wnd(M_PI),
        scale_wnd(0.75) {}


    inline PatLoc sample( const PatLoc& center )
    {
      float lo = center.y - y_wnd;
      if ( lo < 0.0 ) lo = 0.0;
      float hi = center.y + y_wnd;
      if ( hi > rows - 1.0 ) hi = rows - 1.0;
      int y = static_cast<int>( static_cast<float>( rand() ) / RAND_MAX * ( hi - lo ) + lo + 0.5 );

      lo = center.x - x_wnd;
      if ( lo < 0.0 ) lo = 0.0;
      hi = center.x + x_wnd;
      if ( hi > cols - 1.0 ) hi = cols - 1.0;
      int x = static_cast<int>( static_cast<float>( rand() ) / RAND_MAX * ( hi - lo ) + lo + 0.5 );


      float rot = center.rotation + static_cast<float>( rand() ) / RAND_MAX * ( rot_wnd + rot_wnd ) - rot_wnd;

      lo = center.scale - scale_wnd;
      if ( lo < 0.5 ) lo = 0.5;
      hi = center.scale + scale_wnd;
      if ( hi > 2.0 ) hi = 2.0;
      float scale = static_cast<float>( rand() ) / RAND_MAX * ( hi - lo ) + lo;

      return PatLoc( center.id, y, x, rot, scale );
      
    }

    inline void shrink( float alpha = 0.5 )
    {
      y_wnd *= alpha;
      x_wnd *= alpha;
      rot_wnd *= alpha;
      scale_wnd *= alpha;
    }

  };
  

  
  class Geometric
  {
  public:
    int id;
    float dy;
    float dx;
    float rotation;
    float scale;

    /* constructor */
    Geometric ( int id1, float dy1, float dx1, float rot1, float sc1 )
      : id(id1), dy(dy1), dx(dx1), rotation(rot1), scale(sc1) {}

    Geometric() : id(0), dy(0.0f), dx(0.0f), rotation(0.0f), scale(1.0f) {}

    /* ---------- I/O ---------- */
    inline void write( FILE* out )
    {
      fwrite( &id, sizeof(int), 1, out );
      fwrite( &dy, sizeof(float), 1, out );
      fwrite( &dx, sizeof(float), 1, out );
      fwrite( &rotation, sizeof(float), 1, out );
      fwrite( &scale, sizeof(float), 1, out );
    }

    inline void read( FILE* in )
    {
      fread( &id, sizeof(int), 1, in );
      fread( &dy, sizeof(float), 1, in );
      fread( &dx, sizeof(float), 1, in );
      fread( &rotation, sizeof(float), 1, in );
      fread( &scale, sizeof(float), 1, in );
    }

    /* ---------- Operations ---------- */

    /* difference between PatLocs ( a -> b ) */
    static Geometric diff( const PatLoc& a, const PatLoc& b )
    {
      float s = b.scale / a.scale;
      float rotation = b.rotation - a.rotation;
      float scosa = s * cosf( rotation );
      float ssina = s * sinf( rotation );
      float dy = b.y - a.y * scosa - a.x * ssina;
      float dx = b.x + a.x * ssina - a.x * scosa;
      return Geometric( b.id, dy, dx, rotation, s );
    }

    /* apply the transformation on a PatLoc to produce another */
    inline PatLoc apply( const PatLoc&  src ) const
    {
      float s = src.scale * scale;
      float rot = src.rotation + rotation;
      float scosa = scale * cosf( rotation );
      float ssina = scale * sinf( rotation );
      int y1 = static_cast<int>( scosa * src.y + ssina * src.x + dy );
      int x1 = static_cast<int>( -ssina * src.y + scosa * src.x + dx );
      return PatLoc( id, y1, x1, rot, s );
    }

    /* apply the transformation on a standard  to produce another */
    inline PatLoc apply( int y0, int x0 ) const
    {
      float scosa = scale * cosf( rotation );
      float ssina = scale * sinf( rotation );
      int y1 = static_cast<int>( scosa * y0 + ssina * x0 + dy );
      int x1 = static_cast<int>( -ssina * y0 + scosa * x0 + dx );
      return PatLoc( id, y1, x1, rotation, scale );
    }
    
  };

  


  
  struct GeoMap
  {
  private:
    std::vector<std::vector<Geometric> > pool;
    std::vector<int> pos;

  public:
    int rows, cols;

  private:

    GeoMap( const GeoMap& other );
    const GeoMap& operator=( const GeoMap& other );

  public:

    /* ---------- constructor ---------- */
    GeoMap( int h, int w ) : rows(h), cols(w)
    {
      pool.resize( h * w );
      for ( int i=0; i<h*w; i++ ) pool[i].clear();
      pos.resize(h);
      for ( int i=0; i<h; i++ ) pos[i] = i * w;
    }
    
    
    GeoMap( const std::string filename )
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
    GeoMap( GeoMap&& other ) noexcept
    {
      rows = other.rows;
      cols = other.cols;
      pool.swap( other.pool );
      pos.swap( other.pos );
    }

    // Move Assignment
    const GeoMap& operator=( GeoMap&& other )
    {
      rows = other.rows;
      cols = other.cols;
      pool.swap( other.pool );
      pos.swap( other.pos );
      return (*this);
    }
    
  public:
    // I/O operations
    void write( std::string filename )
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
    const std::vector<Geometric>& operator()( int y, int x ) const
    {
      return pool[pos[y]+x];
    }
    
    const std::vector<Geometric>& operator()( int i ) const
    {
      return pool[i];
    }
    
    std::vector<Geometric>& operator[]( int i )
    {
      return pool[i];
    }


    // Merge Operator
    const GeoMap& operator+=( const GeoMap& other )
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





