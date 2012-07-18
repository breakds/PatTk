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

#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/candy.hpp"

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
  // +-------------------------------------------------------------------------------
  template <typename cellType, typename valueType>
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
    Image( Image<cellType,valueType>& img ) {}
    const Image& operator=( Image<cellType,valueType>& img ) {}
    
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
    Image( Image<cellType,valueType>&& img ) : cols(img.cols), rows(img.rows)
    {
      m_size = cols * rows;
      m_cels.swap( img.m_cels );
      m_vals.swap( img.m_vals );
      id = img.id;
      patmask = std::move( img.patmask );
      fullpath.swap( img.fullpath );
    }

    // move assignment
    inline const Image& operator=( Image<cellType,valueType>&& img )
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

      Mask() : height(1), width(1), stride(1)
      {
        mask.resize(1);
        mask[0] = 0;
        cellShift.resize(1);
        cellShift[0] = 0;
        inCellShift.resize(1);
        inCellShift[0] = 0;
      }

      const Mask& operator=( Mask&& other )
      {
        height = other.height;
        width = other.width;
        stride = other.stride;
        mask.swap( other.mask );
        cellShift.swap( other.cellShift );
        inCellShift.swap( other.inCellShift );
        return (*this);
      }
    };

    Mask patmask;

  public:

    class Patch
    {
    private:
      int pos;
    public:
      int y, x, pid;
      const Image<cellType,valueType> &parent;

    public:
      // constructors for Patch
      Patch( const Image<cellType,valueType> &img ) : pos(0), y(0), x(0), pid(img.id), parent(img) {}
      Patch( const Image<cellType,valueType> &img, int y1, int x1 ) : y(y1), x(x1), pid(img.id), parent(img)
      {
        pos = y * parent.cols + x;
      }

      // copy constructor
      Patch( const Patch& patch ) : pos(patch.pos), y(patch.y), x(patch.x), pid(patch.pid), parent(patch.parent) {}
      
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

      // cell selector
      inline const cellType& operator()( const int index ) const
      {
        return parent.GetPatchCell( pos, index );
      }

      // component selector
      inline const typename cellType::type& operator[]( const int index ) const
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
    
    void SetPatchParameter( int height, int width, int stride=1 )
    {
      patmask.width = width;
      patmask.height = height;
      patmask.stride = stride;
      patmask.mask.resize( height * width );
      patmask.cellShift.resize( GetPatchDim() );
      patmask.inCellShift.resize( GetPatchDim() );
      // initialize patmask
      // Assume all the cells have the same length
      int p = 0;
      for ( int i=0, y=0; i<height; i++, y+=stride ) {
        for ( int j=0, x=0; j<width; j++, x+=stride ) {
          patmask.mask[ i * width + j ] = y * cols + x;
          for (int k=0; k<m_cels[0].length; k++ ) {
            patmask.cellShift[p] = y * cols + x;
            patmask.inCellShift[p++] = k;
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

    inline const typename cellType::type& GetPatchComponent( int pos, int index ) const
    {
      return m_cels[pos+patmask.cellShift[index]](patmask.inCellShift[index]);
    }

    inline Patch Spawn( int y, int x ) const
    {
      return Patch( (*this), y, x );
    }
  };


  // +-------------------------------------------------------------------------------
  // | Album, collection of imges.
  // | Serve as owner of images, as well as owner of patches.
  // +-------------------------------------------------------------------------------
  template <typename dataType, typename valueType>
  class Album
  {
  private:
    std::vector< Image<dataType, valueType> > pages;

    // prohibited copy constructor
    Album( const Album<dataType,valueType>& other ) {}

    // prohibited copy assignment
    const Album<dataType,valueType>& operator=( const Album<dataType,valueType>& other ) {}
    
  public:

    Album()
    {
      pages.clear();
    }

    // move constructor
    Album( Album<dataType,valueType>&& other )
    {
      pages.swap( other.pages );
    }

    // move assignment
    const Album<dataType,valueType>& operator=( Album<dataType,valueType>&& other )
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
    inline void push( Image<dataType, valueType>&& img )
    {
      img.id = static_cast<int>( pages.size() );
      pages.push_back( std::move(img) );
    }
    
    // Only provide read-only access to member images
    const Image<dataType, valueType>& operator()( const int index )
    {
      return pages[index];
    }
  };

};






