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
  template <typename dataType>
  class AbstractCell
  {
  private:
    // Prohibit constructor to be called with no argument
    AbstractCell() {}


    // Specializations for trace()
    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,int>::value>::type
                       __attribute__((__unused__)) *padding=0 )
    {
      printf( "%d", (*this)[0] );
      for ( int i=1; i<length; i++ ) {
        printf( " %d", (*this)[i] );
      }
    }

    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,unsigned char>::value>::type
                       __attribute__((__unused__)) *padding=0 )
    {
      printf( "%hhu", (*this)[0] );
      for ( int i=1; i<length; i++ ) {
        printf( " %hhu", (*this)[i] );
      }
    }

    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,double>::value>::type
                       __attribute__((__unused__)) *padding=0 )
    {
      printf( "%.2lf", (*this)[0] );
      for ( int i=1; i<length; i++ ) {
        printf( " %.2lf", (*this)[i] );
      }
    }
    
    template <typename T>
    inline void trace( typename std::enable_if< std::is_same<T,float>::value>::type
                       __attribute__((__unused__)) *padding=0 )
    {
      printf( "%.2f", (*this)[0] );
      for ( int i=1; i<length; i++ ) {
        printf( " %.2f", (*this)[i] );
      }
    }

    
  public:
    typedef dataType type;
    
    int length;
    
    // Constructor
    AbstractCell( const int len ) : length(len) {}
    
    // non-const selector of elements
    virtual dataType& operator[]( const int index ) = 0;

    // const selector of elements
    virtual const dataType& operator()( const int index ) const = 0;

    // print the cell to screen
    inline void trace()
    {
      trace<dataType>();
    }

    void Summary()
    {
      printf( "#(" );
      trace();
      printf( ")  len: %d\n", length );
    }
    
  };


  // Variadic Concatenation of CellTypes
  // Should also be a valid cell type
  template <typename dataType, typename... cellTypes> class Concat {};

  template <typename dataType> class Concat<dataType>
  {
  public:
    void trace() {}
  };
  
  template <typename dataType, typename Head, typename... Tail>
  class Concat< dataType, Head, Tail... > : private Concat<dataType, Tail...>
  {
    static_assert( std::is_same< dataType, typename Head::type >::value,
                   "Concat and Head do not share the same data type. " );
  protected:
    Head m_head;

  public:
    static const int num = 1 + count<Tail...>::value;
    
    Concat() {}

    Concat( const Head& v, const Tail&... vtail ) : Concat<dataType, Tail...>(vtail...), m_head(v) {}

    // copy constructor
    template <typename ...cellTypes>
    Concat( const Concat<dataType, cellTypes...>& other ) :
      Concat<dataType, Tail...>(other.tail()), m_head(other.head()) {}

    // move constructor
    template <typename ...cellTypes>
    Concat( Concat<dataType, cellTypes...>&& other ) :
      Concat<dataType, Tail...>(std::move(other.tail())), m_head(std::move(other.head())) {}


    // Access head and tail (tail = instance of base class)
    Head& head() { return m_head; }
    const Head& head() const { return m_head; }
    // There is a trick that since Concat<Tail...> is the base class,
    // there is going to be a implicit conversion
    Concat<dataType, Tail...>& tail() { return *this; }
    const Concat<dataType, Tail...>& tail() const {return *this; }
    
    void trace() {
      head().trace();
      if ( num > 1 ) {
        printf( " " );
      }
      tail().trace();
    }
    
    void Summary() {
      printf( "count: %d\n", num );
      printf( "#(" );
      trace();
      printf( ")\n" );
    }
  };
  
  
};






