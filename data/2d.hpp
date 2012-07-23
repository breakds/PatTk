/*********************************************************************************
 * File: 2d.hpp
 * Description: 2 dimensional cell-based data structure model for features
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:15:22 CDT 2012
 *********************************************************************************/

#pragma once
#include <type_traits>
#include <vector>
#include <tuple>
#include <string>
#include <cmath>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/candy.hpp"


#define _USE_MATH_DEFINES
#define vertical_x -horiz_y
#define vertical_y horiz_x

namespace PatTk
{

  // +-------------------------------------------------------------------------------
  // | Abstract Template Class for Cell
  // | Example: 1. class A : AbstractCell<unsigned char>
  // |          2. template <typename dataType>
  // |             class B : AbstractCell<dataType>
  // +-------------------------------------------------------------------------------
  template <typename dataType>
  class AbstractCell
  {
    static_assert( std::is_integral<dataType>::value || std::is_floating_point<dataType>::value,
                   "Non numeric type is not allowed as dataType in AbstractCell." );
  private:
    // Prohibit constructor to be called with no argument
    AbstractCell() {}


    // Specializations for trace()
    // Those specializations print the content of the cell
    // as a vector, with no "\n" or "\r"
    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,int>::value>::type
                       __attribute__((__unused__)) *padding=0 ) const
    {
      printf( "%d", (*this)(0) );
      for ( int i=1; i<length; i++ ) {
        printf( " %d", (*this)(i) );
      }
    }

    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,unsigned char>::value>::type
                       __attribute__((__unused__)) *padding=0 ) const
    {
      printf( "%hhu", (*this)(0) );
      for ( int i=1; i<length; i++ ) {
        printf( " %hhu", (*this)(i) );
      }
    }

    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,double>::value>::type
                       __attribute__((__unused__)) *padding=0 ) const
    {
      printf( "%.2lf", (*this)(0) );
      for ( int i=1; i<length; i++ ) {
        printf( " %.2lf", (*this)(i) );
      }
    }
    
    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,float>::value>::type
                       __attribute__((__unused__)) *padding=0 ) const
    {
      printf( "%.2f", (*this)(0) );
      for ( int i=1; i<length; i++ ) {
        printf( " %.2f", (*this)(i) );
      }
    }

    
  public:
    // The type of the data.  Example: if B is derived from
    // AbstractCell<int>, then B::type is int. Probably need
    // to use "typename B::type" to be explicit.
    typedef dataType type;


    static const bool RotationSensitive = false;
    

    // The length of content in the cell
    int length;
    
    // Constructor
    AbstractCell( const int len ) : length(len) {}
    
    // non-const selector of elements
    virtual dataType& operator[]( const int index ) = 0;

    // const selector of elements
    virtual const dataType& operator()( const int index ) const = 0;

    
    void trace() const
    {
      trace<dataType>();
    }

    // print the cell to screen
    void Summary() const
    {
      printf( "#(" );
      trace();
      printf( ")  len: %d\n", length );
    }
    
  };

  
  // +-------------------------------------------------------------------------------
  // | Container class for images
  // | Example: Image< HistCell<unsigned char>, vector<int>  >
  //            Image< HistCell<unsigned char>, int, true >
  // +-------------------------------------------------------------------------------
  template <typename cellType, typename valueType, bool lite = true >
  class Image
  {


    // cellType must be a valid AbstractCell or a concat of them.
    static_assert( std::is_base_of< AbstractCell<typename cellType::type>, cellType >::value,
                   "cellType is not a valid cell type. (does not derive from AbstractCell." );
  private:
    // containers:
    std::vector<cellType> m_cels;
    std::vector<valueType> m_vals;
    // properties:
    int m_size;

  private:
    // prohibit from calling copy constructor/assignment operator
    Image( Image<cellType,valueType,lite>& img ) {}
    const Image& operator=( Image<cellType,valueType,lite>& img ) {}
    
  public:

    int cols, rows;
    int id;
    std::string fullpath;

    /// Consturctors:
    
    Image() : m_cels(), m_vals(), cols(0), rows(0), id(0), fullpath("") {}

    Image( int h, int w ) : cols(w), rows(h), id(0), fullpath("")
    {
      m_size = h * w;
      m_cels.resize( m_size );
      m_vals.resize( m_size );
    }

    // move constructor
    Image( Image<cellType,valueType,lite>&& img ) : cols(img.cols), rows(img.rows)
    {
      m_size = cols * rows;
      m_cels.swap( img.m_cels );
      m_vals.swap( img.m_vals );
      id = img.id;
      patmask = std::move( img.patmask );
      fullpath.swap( img.fullpath );
    }

    // move assignment
    inline const Image& operator=( Image<cellType,valueType,lite>&& img )
    {
      cols = img.cols;
      rows = img.rows;
      m_size = cols * rows;
      m_cels.swap( img.m_cels );
      m_vals.swap( img.m_vals );
      patmask = std::move( img.patmask );
      id = img.id;
      return (*this);
    }


    
    // use operator() to access the cell of coordinates (y,x)
    inline const cellType& operator()( const int y, const int x ) const
    {
      return m_cels[ y * cols + x ];
    }

    // use operator() to access the cell of index i
    inline const cellType& operator()( const int i ) const
    {
      return m_cels[i];
    }

    // operator[] to access non-const cell by index i
    inline cellType& operator[]( const int i )
    {
      return m_cels[i];
    }
    


    
    /// Selectors/Setters:
    
    // only rvalue semantic is provided for altering the content of cells
    inline void setCell( const int y, const int x, cellType&& cell )
    {
      // will call rvalue assignment automatically
      m_cels[ y * cols + x ] = cell;
    }
    inline void setCell( const int i, cellType&& cell )
    {
      // will call rvalue assignment automatically
      m_cels[i] = cell;
    }

    
    // access the val of coordinates (y,x)
    inline const valueType& getVal( const int y, const int x ) const
    {
      return m_vals[ y * cols + x ];
    }

    // access the val of coordinates i
    inline const valueType& getVal( const int i ) const
    {
      return m_vals[i];
    }
    
    void setFullPath( const std::string& filename )
    {
      fullpath = filename;
    }
    
    // only rvalue semantic is provided for altering the values
    inline void setVal( const int y, const int x, valueType&& val )
    {
      // will call rvalue assignment automatically
      m_vals[ y * cols + x ] = val;
    }
    inline void setVal( const int i, valueType&& val )
    {
      // will call rvalue assignment automatically
      m_vals[i] = val;
    }


    /// Properties:
    
    // whether image is empty
    inline bool empty() const
    {
      return !( rows && cols );
    }

    inline int size() const
    {
      return m_size;
    }


    /// Debug Utilities
    void Summary() const
    {
      printf( "Image of %d x %d (%ld).\n", cols, rows, m_vals.size()  );
    }


  private:
    /// Patch Utilities
    static int default_patch_height;
    static int default_patch_width;

    struct Mask
    {
    public:
      int height, width, stride;
      std::vector<int> mask; // the offset in the Image
      std::vector<int> cellShift; // the offset in the mask
      std::vector<int> inCellShift; // the offset in a cell
      std::vector<int> x_offset;
      std::vector<int> y_offset;

      Mask() : height(1), width(1), stride(1)
      {
        mask.resize(1);
        mask[0] = 0;
        cellShift.resize(1);
        cellShift[0] = 0;
        inCellShift.resize(1);
        inCellShift[0] = 0;
        x_offset.resize(1);
        x_offset[0] = 0;
        y_offset.resize(1);
        y_offset[0] = 0;
        
      }

      const Mask& operator=( Mask&& other )
      {
        height = other.height;
        width = other.width;
        stride = other.stride;
        mask.swap( other.mask );
        cellShift.swap( other.cellShift );
        inCellShift.swap( other.inCellShift );
        x_offset.swap( other.x_offset );
        y_offset.swap( other.y_offset );
        return (*this);
      }
    };

    Mask patmask;

  public:


    template <bool T, typename enabled = void>
    class AttachedPatch {};

    
    /// Specialization: Lite Weighted Patch
    
    template <bool T>
    class AttachedPatch<T,typename std::enable_if<T>::type>
    {
    private:
      int pos;
    public:
      int y, x, pid;
      const Image<cellType,valueType,lite> &parent;

    public:
      // constructors for Patch
      AttachedPatch( const Image<cellType,valueType,lite> &img ) : pos(0), y(0), x(0), pid(img.id), parent(img) {}
      AttachedPatch( const Image<cellType,valueType,lite> &img, int y1, int x1 ) : y(y1), x(x1), pid(img.id), parent(img)
      {
        pos = y * parent.cols + x;
      }

      // copy constructor
      AttachedPatch( const AttachedPatch<T>& patch )
        : pos(patch.pos), y(patch.y), x(patch.x), pid(patch.pid), parent(patch.parent) {}
      
      inline bool isValid() const
      {
        if ( 0 <= x && x + parent.GetPatchWidth() <= parent.cols &&
             0 <= y && y + parent.GetPatchHeight() <= parent.rows ) {
          return true;
        }
        return false;
      }

      inline int cellNum() const
      {
        return parent.GetPatchCellNum();
      }

      inline int dim() const
      {
        return parent.GetPatchDim();
      }

      inline double GetScale() const
      {
        return 1.0;
      }

      inline double GetRotation() const
      {
        return 0.0;
      }

      // cell selector
      inline const cellType& operator()( const int index ) const
      {
        return parent.GetPatchCell( pos, index );
      }

      // component selector
      inline const typename cellType::type operator[]( const int index ) const
      {
        return parent.GetPatchComponent( pos, index );
      }

      inline void Summary() const
      {
        printf( "Patch %d:(%d,%d) | dimension %d | %d cells.\n", pid, y, x, dim(), cellNum() );
      }

      inline void trace() const
      {
        for ( int i=0; i<cellNum(); i++ ) {
          if ( i > 0 ) {
            printf( " | " );
          }
          (*this)(i).trace();
        }
      }
      
    };



    /// Specialization: Heavy Weighted Patch
    template <bool T>
    class AttachedPatch<T,typename std::enable_if<!T>::type>
    {
    private:
      double horiz_x, horiz_y; 
      // ( horiz_x, horiz_y ) forms the horizontal vector of length "cell side"
      // ( -horiz_y, horiz_x) turns out to be the vertical vector of same length.
    public:
      double y, x;
      int pid;
      double scale, rotation; // rotation is clockwise and in DEGREE
      const Image<cellType,valueType,lite> &parent;

    public:
      // constructors for Patch
      AttachedPatch( const Image<cellType,valueType,lite> &img )
        : y(0), x(0), pid(img.id), scale(1.0), rotation(0.0),
          parent(img)
      {
        horiz_x = 1.0;
        horiz_y = 0.0;
      }

      AttachedPatch( const Image<cellType,valueType,lite> &img, double y1, double x1 )
        : y(y1), x(x1), pid(img.id),
          scale(1.0), rotation(0.0),
          parent(img)
      {
        horiz_x = 1.0;
        horiz_y = 0.0;
      }



      AttachedPatch( const Image<cellType,valueType,lite> &img, double y1, double x1, double scl, double rot )
        : y(y1), x(x1), pid(img.id),
          scale(scl), rotation(rot),
          parent(img)
      {
        double tmp = rotation * M_PI / 180.0;
        horiz_x = cos( tmp ) / scale;
        horiz_y = sin( tmp ) / scale;
      }


      
      // copy constructor
      AttachedPatch( const AttachedPatch<T>& patch )
        : y(patch.y), x(patch.x), pid(patch.pid),
          scale(patch.scale), rotation(patch.scale),
          parent(patch.parent) {}
      

      inline bool isValid() const
      {
        double x1 = x + parent.GetPatchWidth() * horiz_x;
        double x2 = x + parent.GetPatchHeight() * vertical_x;
        double x3 = x1 + parent.GetPatchHeight() * vertical_x;

        double y1 = y + parent.GetPatchWidth() * horiz_y;
        double y2 = y + parent.GetPatchHeight() * vertical_y;
        double y3 = y1 + parent.GetPatchHeight() * vertical_y;

        double top = INF( y, y1, y2 ,y3 );
        double bottom = SUP( y, y1, y2 ,y3 );

        double left = INF( x, x1, x2 ,x3 );
        double right = SUP( x, x1, x2 ,x3 );

        if ( 0.0 <= left && right <= parent.cols &&
             0.0 <= top && bottom <= parent.rows ) {
          return true;
        }
        return false;
      }
      
      inline int cellNum() const
      {
        return parent.GetPatchCellNum();
      }
      
      inline int dim() const
      {
        return parent.GetPatchDim();
      }

      // cell selector
      // inline const cellType& operator()( const int index ) const
      // {
      //   return parent.GetPatchCell( pos, index );
      // }
      
      // component selector
      inline const typename cellType::type operator[]( const int index ) const
      {
        return parent.GetPatchComponent<cellType::RotationSensitive>( y, x, rotation, horiz_y, horiz_x, index );
      }

      inline void Summary() const
      {
        printf( "Patch %d:(%.2lf,%2lf) | dimension %d | %d cells.\n", pid, y, x, dim(), cellNum() );
      }

      inline void trace() const
      {
        printf( "trace() is not currently supported in Heavy Weighted Patch." );
      }
      
    };

  public:
    typedef AttachedPatch<lite> Patch;

    inline const typename cellType::type Interpolate( double x, double y, int i ) const
    {
      // TODO: shfit
      int y0 = static_cast<int>(y);
      int x0 = static_cast<int>(x);
      int y1 = y0 + 1;
      int x1 = x0 + 1;
      double frac_y = y - y0;
      double frac_x = x - x0;
      double left = (*this)(y0,x0)(i) * (1.0 - frac_y) + (*this)(y1,x0)(i) * frac_y;
      double right = (*this)(y0,x1)(i) * (1.0 - frac_y) + (*this)(y1,x1)(i) * frac_y;
      return left * (1.0 - frac_x) + right * frac_x;
    }

    void SetPatchParameter( int height, int width, int stride=1 )
    {
      patmask.width = width;
      patmask.height = height;
      patmask.stride = stride;
      patmask.mask.resize( height * width );
      patmask.cellShift.resize( GetPatchDim() );
      patmask.inCellShift.resize( GetPatchDim() );
      patmask.x_offset.resize( GetPatchDim() );
      patmask.y_offset.resize( GetPatchDim() );
      // initialize patmask
      // Assume all the cells have the same length
      int p = 0;
      for ( int i=0, y=0; i<height; i++, y+=stride ) {
        for ( int j=0, x=0; j<width; j++, x+=stride ) {
          patmask.mask[ i * width + j ] = y * cols + x;
          for (int k=0; k<m_cels[0].length; k++, p++ ) {
            patmask.cellShift[p] = y * cols + x;
            patmask.inCellShift[p] = k;
            patmask.x_offset[p] = j;
            patmask.y_offset[p] = i;
          }
        }
      }
    }

    inline int GetPatchWidth() const
    {
      return 1 + ( patmask.width - 1 ) * patmask.stride;
    }

    inline int GetPatchHeight() const
    {
      return 1 + ( patmask.height - 1 ) * patmask.stride;
    }

    inline int GetPatchCellNum() const
    {
      return patmask.width * patmask.height;
    }

    inline int GetPatchDim() const
    {
      return patmask.width * patmask.height * m_cels[0].length;
    }

    inline const cellType& GetPatchCell( int pos, int offsetIdx ) const
    {
      return m_cels[pos+patmask.mask[offsetIdx]];
    }

    inline const typename cellType::type GetPatchComponent( int pos, int index ) const
    {
      return m_cels[pos+patmask.cellShift[index]](patmask.inCellShift[index]);
    }

    template <bool T>
    inline const typename cellType::type GetPatchComponent( double y, double x, double rotation, double horiz_y,
                                                            double horiz_x, int index,
                                                            typename std::enable_if<T>::type
                                                            __attribute__((__unused__)) *padding=0 ) const
    {
      return Iterpolate( y + patmask.y_offset[index] * vertical_y + patmask.x_offset[index] * horiz_y,
                         x + patmask.y_offset[index] * vertical_x + patmask.x_offset[index] * horiz_x,
                         patmask.inCellShift[index] );
    }

    template <bool T>
    inline const typename cellType::type GetPatchComponent( double y, double x, double rotation, double horiz_y,
                                                            double horiz_x, int index,
                                                            typename std::enable_if<!T>::type
                                                            __attribute__((__unused__)) *padding=0 ) const
    {
      static const double ang_unit = 1.0 / 180.0;
      int base = index - patmask.inCellShift[index];
      double offset = patmask.inCellShift[index] + rotation * ang_unit;
      if ( (*this)(0).length <= offset ) {
        offset -= (*this)(0).length;
      } else if ( 0 > offset ) {
        offset += (*this)(0).length;
      }

      int pos0 = static_cast<int>( offset );
      double ratio = offset - pos0;
      int pos1 = ( pos0 + 1 == (*this)(0).length ) ? 0 : pos0 + 1;
      pos0 += base;
      pos1 += base;

      typename cellType::type a = Interpolate( y + patmask.y_offset[pos0] * vertical_y +
                                              patmask.x_offset[pos0] * horiz_y,
                                              x + patmask.y_offset[pos0] * vertical_x +
                                              patmask.x_offset[pos0] * horiz_x,
                                              patmask.inCellShift[pos0] );
      typename cellType::type b = Interpolate( y + patmask.y_offset[pos1] * vertical_y +
                                              patmask.x_offset[pos1] * horiz_y,
                                              x + patmask.y_offset[pos1] * vertical_x +
                                              patmask.x_offset[pos1] * horiz_x,
                                              patmask.inCellShift[pos1] );
      return static_cast<typename cellType::type>( a * ( 1.0 - ratio ) + b * ratio );
    }


    template<bool T = lite>
    inline Patch Spawn( int y, int x,
                        typename std::enable_if<T>::type __attribute__((__unused__)) *padding=0 ) const
    {
      return Patch( (*this), y, x );
    }

    template<bool T = lite>
    inline Patch Spawn( double y, double x, double scl, double rot,
                        typename std::enable_if<!T>::type __attribute__((__unused__)) *padding=0 ) const
    {
      return Patch( (*this), y, x, scl, rot );
    }

  };


  // +-------------------------------------------------------------------------------
  // | Album, collection of imges.
  // | Serve as owner of images, as well as owner of patches.
  // +-------------------------------------------------------------------------------
  template <typename dataType, typename valueType, bool lite = true>
  class Album
  {
  private:
    std::vector< Image<dataType, valueType, lite> > pages;

    // prohibited copy constructor
    Album( const Album<dataType,valueType, lite>& other ) {}

    // prohibited copy assignment
    const Album<dataType,valueType, lite>& operator=( const Album<dataType,valueType, lite>& other ) {}
    
  public:

    Album()
    {
      pages.clear();
    }

    // move constructor
    Album( Album<dataType,valueType, lite>&& other )
    {
      pages.swap( other.pages );
    }

    // move assignment
    const Album<dataType,valueType, lite>& operator=( Album<dataType,valueType, lite>&& other )
    {
      pages.swap( other.pages );
      return (*this);
    }

    // size()
    int size() const
    {
      return static_cast<int>( pages.size() );
    }

    // push() has side effect. It is destructive as it steals the
    // Image that img refer to.
    inline void push( Image<dataType, valueType, lite>&& img )
    {
      img.id = static_cast<int>( pages.size() );
      pages.push_back( std::move(img) );
    }
    
    // Only provide read-only access to member images
    const Image<dataType, valueType, lite>& operator()( const int index )
    {
      return pages[index];
    }

    Image<dataType, valueType, lite>& back()
    {
      return pages.back();
    }

    // Set patch parameters, applied to all the images
    void SetPatchParameter( int height, int width, int stride=1 )
    {
      for ( auto& every : pages ) {
        every.SetPatchParameter( height, width, stride );
      }
    }
  };
  
};






