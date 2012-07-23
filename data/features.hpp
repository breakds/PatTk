/*********************************************************************************
 * File: features.hpp
 * Description: implementation of several commonly used feature cells
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:15:22 CDT 2012
 *********************************************************************************/

#pragma once
#include <type_traits>
#include <cassert>
#include "2d.hpp"


using std::vector;


namespace PatTk
{

  template <typename dataType>
  class HistCell : public AbstractCell<dataType>
  {
  public:
    static const bool RotationSensitive = false;
  private:
    vector<dataType> h;
  public:

    HistCell() : AbstractCell<dataType>(0) {}
    HistCell( int n ) : AbstractCell<dataType>(n) {}
    HistCell( vector<dataType>&& content ) : AbstractCell<dataType>( static_cast<int>( content.size() ) )
    {
      h = content;
    }
    
    HistCell( const HistCell& other ) : AbstractCell<dataType>(other.length)
    {
      h = other.h;
    }

    HistCell( HistCell&& other ) : AbstractCell<dataType>(other.length)
    {
      h = std::move( other.h );
    }


    // move assignment
    const HistCell& operator=( HistCell&& other )
    {
      this->length = other.length;
      h.swap( other.h );
      return (*this);
    }

    // copy assignment
    const HistCell& operator=( const HistCell& other )
    {
      this->length = other.length;
      h = other.h;
      return (*this);
    }

    // extra operator
    const HistCell& operator+=( const HistCell& other )
    {
      assert( this->length == other.length );
      for ( int i=0; i<this->length; i++ ) {
        h[i] += other.h[i];
      }
      return (*this);
    }


    const HistCell& operator-=( const HistCell& other )
    {
      assert( this->length == other.length );
      for ( int i=0; i<this->length; i++ ) {
        h[i] -= other.h[i];
      }
      return (*this);
    }


    const HistCell& operator/=( const double divisor )
    {
      for ( int i=0; i<this->length; i++ ) {
        h[i] /= divisor;
      }
      return (*this);
    }

    
    inline void reset( const int s )
    {
      this->length = s;
      h.resize(s);
      for ( auto& ele : h ) ele = 0;
    }

    inline dataType& operator[]( const int index )
    {
      return h[index];
    }
    
    inline const dataType& operator()( const int index ) const
    {
      return h[index];
    }

    template <typename array>
    inline void NormalizeToUchar( array& a )
    {
      int bins = this->length;
      double sum = 0;

      for ( auto& ele : h ) sum += ele;

      
      if ( sum < 1e-5 ) {
        for ( int i=0; i<bins; i++ ) {
          a[i] = 0;
        }
        return ;
      }
      uint index[bins];
      double score[bins];
      int remainder = 128;
      for ( int i=0; i<bins; i++ ) {
        score[i] = h[i] / sum * 128;
        a[i] = static_cast<unsigned char>( score[i] );
        score[i] -= a[i];
        remainder -= a[i];
        index[i] = i;
      }

      // Sort
      for ( int i=0; i<bins-1; i++ ) {
        for ( int j=i+1; j<bins; j++ ) {
          if ( score[index[i]] < score[index[j]] ) {
            uint tmp = index[i];
            index[i] = index[j];
            index[j] = tmp;
          }
        }
      }
  
      uint k = 0;
      while ( remainder > 0 ) {
        remainder--;
        a[index[k++]]++;
      }
    }
    
  };



  class HoGCell : public HistCell<unsigned char>
  {
  public:
    static const int RotationSensitive = true;

    static inline unsigned char resample( const unsigned char a, const unsigned char b, const double ratio )
    {
      return static_cast<unsigned char>( a * ( 1.0 - ratio ) + b * ratio );
    }
  };
  





  
  // Container for an CIE L*a*b* cell (raw Lab color feature)
  class LabCell : public AbstractCell<unsigned char>
  {
  public:
    static const bool RotationSensitive = false;
  private:
    unsigned char lab[3];
  public:
    LabCell() : AbstractCell(3) {}
    LabCell( const unsigned char l, const unsigned char a, const unsigned char b ) : AbstractCell(3)
    {
      lab[0] = l; // 0 for luminance channel
      lab[1] = a; // 1 for "a" channel
      lab[2] = b; // 2 for "b" channel
    }

    // copy constructor
    LabCell( const LabCell& other ) : AbstractCell(3)
    {
      lab[0] = other.lab[0];
      lab[1] = other.lab[1];
      lab[2] = other.lab[2];
    }

    // move constructor
    LabCell( LabCell&& other ) : AbstractCell(3)
    {
      lab[0] = other.lab[0];
      lab[1] = other.lab[1];
      lab[2] = other.lab[2];
    }

    // move assignment
    const LabCell& operator=( LabCell&& other )
    {
      lab[0] = other.lab[0];
      lab[1] = other.lab[1];
      lab[2] = other.lab[2];
      return (*this);
    }

    // copy assignment
    const LabCell& operator=( LabCell& other )
    {
      lab[0] = other.lab[0];
      lab[1] = other.lab[1];
      lab[2] = other.lab[2];
      return (*this);
    }

    
    inline unsigned char& operator[]( const int index )
    {
      assert( index < length );
      return lab[index];
    }

    inline const unsigned char& operator()( const int index ) const
    {
      assert( index < length );
      return lab[index];
    }
  };


  // +-------------------------------------------------------------------------------
  // | Integral Operations for Image
  // | Example: IntegralImage( img );
  // +-------------------------------------------------------------------------------
  template <typename cellType, typename valueType>
  void IntegralImage( Image<cellType,valueType>& img, int wndSize ) 
  {
    typedef typename cellType::type data_t;
    static_assert( std::is_same<HistCell<data_t>, cellType>::value,
                   "IntergralImage() - only supports HistCell." );

    cellType tmp[img.rows][img.cols];
    for ( int i=0; i<img.rows; i++ ) {
      for ( int j=0; j<img.cols; j++ ) {
        tmp[i][j] = img(i,j);
      }
    }

    for ( int i=0; i<img.rows; i++ ) {
      for ( int j=1; j<img.cols; j++ ) {
        tmp[i][j] += tmp[i][j-1];
      }
    }

    for ( int j=0; j<img.cols; j++ ) {
      for ( int i=1; i<img.rows; i++  ) {
        tmp[i][j] += tmp[i-1][j];
      }
    }


    int neg = wndSize >> 1;
    int pos = wndSize - neg - 1; 


    for ( int i=0; i<img.rows; i++ ) {
      for ( int j=0; j<img.cols; j++ ) {
        int bottom = ( i + pos < img.rows ) ? i + pos : img.rows - 1;
        int right = ( j + pos < img.cols ) ? j + pos : img.cols - 1;
        int top = i - neg;
        int left = j - neg;
        int k = i * img.cols + j;
        img[k] = tmp[bottom][right];

        if ( 0 < top && 0 < left ) {
          img[k] += tmp[top-1][left-1];
        }
        
        if ( 0 < top ) {
          img[k] -= tmp[top-1][right];
        } else {
          top = 0;
        }
        
        if ( 0 < left ) {
          img[k] -= tmp[bottom][left-1];
        } else {
          left = 0;
        }

        img[k] /= ( (bottom - top + 1) * (right - left + 1) );
      }
    }
    
  }
  
};

