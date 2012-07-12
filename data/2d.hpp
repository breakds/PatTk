/*********************************************************************************
 * File: 2d.hpp
 * Description: 2 dimensional cell-based data structure model for features
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:15:22 CDT 2012
 *********************************************************************************/

#pragma once
#include <type_traits>
#include <vector>

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
  // | Variadic Concatenation of CellTypes
  // | Example: concat< unsigned char, A, B >
  // +-------------------------------------------------------------------------------
  template <typename dataType, typename... cellTypes> class concat {};

  // Base Class, no variadic template arguments
  template <typename dataType> class concat<dataType>
  {
  public:
    typedef dataType type;
    int length;


    const dataType& operator()( const int index ) const
    {
      static int tmp = 0;
      return ((dataType) tmp);
    }
    concat() : length(0) {}
    void trace() {}
    
  };


  // iteratively define variadic templated class
  template <typename dataType, typename Head, typename... Tail>
  class concat< dataType, Head, Tail... > : private concat<dataType, Tail...>
  {
    // if one element does not have the same datatype, fail the compilation
    static_assert( std::is_same< dataType, typename Head::type >::value,
                   "concat and Head do not share the same data type. " );
  protected:
    Head m_head;
    
  public:
    typedef dataType type;

    // number of elements
    static const int num = 1 + count<Tail...>::value;

    // length of the whole concatenated cell
    int length;
    
    concat() {}

    concat( const Head& v, const Tail&... vtail ) : concat<dataType, Tail...>(vtail...), m_head(v) 
    {
      length = v.length + tail().length;
    }

    // copy constructor
    template <typename ...cellTypes>
    concat( const concat<dataType, cellTypes...>& other ) :
      concat<dataType, Tail...>(other.tail()), m_head(other.head()), length(other.length) {}

    // move constructor
    template <typename ...cellTypes>
    concat( concat<dataType, cellTypes...>&& other ) :
      concat<dataType, Tail...>(std::move(other.tail())), m_head(std::move(other.head())), length(other.length) {}


    // index accessor
    dataType& operator[]( const int index )
    {
      if ( index >= head().length ) {
        return tail()[index-head().length];
      }
      return head()[index];
    }

    
    const dataType& operator()( const int index ) const
    {
      if ( index >= head().length ) {
        const dataType& tmp = tail().number();
        return tmp;
      }
      return head()(index);
    }
    

    // Access head and tail (tail = instance of base class)
    Head& head() { return m_head; }
    const Head& head() const { return m_head; }
    // There is a trick that since concat<Tail...> is the base class,
    // there is going to be a implicit conversion
    concat<dataType, Tail...>& tail() { return *this; }
    const concat<dataType, Tail...>& tail() const {return *this; }

    // aux function for Summary()
    void trace() {
      head().trace();
      if ( num > 1 ) {
        printf( " | " );
      }
      tail().trace();
    }

    // print the content out to the screen
    void Summary() {
      printf( "#(" );
      trace();
      printf( ")  len: %d\n", length );
    }
  };

  
  // +-------------------------------------------------------------------------------
  // | Container class for images
  // | Example: Image< concat< LabCell, HistCell<unsigned char> > >
  // +-------------------------------------------------------------------------------
  template <typename cellType, typename valueType>
  class Image
  {


    // cellType must be a valid AbstractCell or a concat of them.
    static_assert( std::is_base_of< AbstractCell<typename cellType::type>, cellType >::value ||
                   std::is_base_of< concat<typename cellType::type>, cellType >::value,
                   "cellType is not a valid cell type. (does not derive from AbstractCell or concat." );
    
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

    /// Consturctors:
    
    Image() : cols(0), rows(0), m_cels(), m_vals() {}

    Image( int h, int w ) : cols(w), rows(h)
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
      patmask = std::move( img.patmask );
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
      printf( "Image of %d x %d.\n", cols, rows );
    }


  private:
    /// Patch Utilities
    static int default_patch_height;
    static int default_patch_width;

    struct Mask
    {
    public:
      int height, width, stride;
      std::vector<int> mask;

      Mask() : height(1), width(1), stride(1)
      {
        mask.resize(1);
        mask[0] = 0;
      }

      const Mask& operator=( Mask&& other )
      {
        height = other.height;
        width = other.width;
        stride = other.stride;
        mask.swap( other.mask );
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
      int y, x;
      const Image<cellType,valueType> &parent;

      // constructors for Patch
      Patch( const Image<cellType,valueType> &img ) : pos(0), y(0), x(0), parent(img) {}
      Patch( const Image<cellType,valueType> &img, int y1, int x1 ) : y(y1), x(x1), parent(img)
      {
        pos = y * parent.cols + x;
      }
      Patch( const Patch& patch ) : pos(patch.pos), y(patch.y), x(patch.x), parent(patch.parent) {}
      
      inline bool isValid() const
      {
        if ( 0 <= x && x + parent.GetPatchWidth() <= parent.cols &&
             0 <= y && y + parent.GetPatchHeight() <= parent.rows ) {
          return true;
        }
        return false;
      }

      inline int dim() const
      {
        return parent.GetPatchDim();
      }

      inline const cellType& operator()( const int index ) const
      {
        return parent.GetPatchCell( pos, index );
      }
    };
    
    void SetPatchParameter( int height, int width, int stride=1 )
    {
      patmask.width = width;
      patmask.height = height;
      patmask.stride = stride;
      patmask.mask.resize( height * width );
      for ( int i=0, y=0; i<height; i++, y+=stride ) {
        for ( int j=0, x=0; j<width; j++, x+=stride ) {
          patmask.mask[ i * width + j ] = y * cols + x;
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

    inline int GetPatchDim() const
    {
      return patmask.width * patmask.height;
    }

    inline const cellType& GetPatchCell( int pos, int offsetIdx ) const
    {
      return m_cels[pos+patmask.mask[offsetIdx]];
    }

    inline Patch Spawn( int y, int x ) const
    {
      return Patch( (*this), y, x );
    }
  };
};






