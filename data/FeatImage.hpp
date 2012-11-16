/*********************************************************************************
 * File: FeatImage.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Provide the Feature Image Class that represent 2d image with each
 *              pixel storing an array. The idea is that concatenation of a set of
 *              such arrays within a window will give the feature descriptor vector
 *              of this window.
 *********************************************************************************/

#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "LLPack/utils/SafeOP.hpp"
#include "vector.hpp"

namespace PatTk
{
  template <typename dataType>
  class FeatImage
  {
  private:
    /* ---------- data ---------- */
    std::vector<dataType> data;
  public:
    /* ---------- properties ---------- */
    int rows, cols;
    int id;
    int dimCell;

  private:
    /* ---------- prohibited methods ---------- */
    FeatImage( FeatImage<dataType>& other ) {}
    const FeatImage<dataType>& operator=( FeatImage<dataType> &other ) {}
    
  public:

    /* ---------- constructor/desctructor/assignment ---------- */

    /* default constructor: zero sized image */
    FeatImage() : data(), rows(0), cols(0), id(-1), dimCell(0) {}

    /* empty image constructor */
    FeatImage( int h, int w, int dim ) : data(h*w*dim,0), rows(h), cols(w), id(-1), dimCell(dim) {}

    /* move constructor */
    FeatImage( FeatImage<dataType> &&other ) : rows(other.rows), cols(other.cols), id(other.id),
                                               dimCell(other.dimCell)
    {
      data.swap( other.data );
    }

    /* move assignment */
    inline const FeatImage<dataType>& operator=( FeatImage<dataType> &&other )
    {
      rows = other.rows;
      cols = other.cols;
      id = other.id;
      dimCell = other.dimCell;
      data.swap( other.data );
    }

    /* clone */
    inline FeatImage<dataType> clone()
    {
      FeatImage<dataType> re;
      re.rows = rows;
      re.cols = cols;
      re.id = id;
      re.dimCell = dimCell;
      re.data = data;
      return re;
    }
    
    /* ---------- Cell Accessors ---------- */

    inline const dataType* operator()( const int y, const int x ) const
    {
      return (&data[0]) + ( y * cols + x ) * dimCell;
    }

    inline const dataType* operator()( const int i ) const
    {
      return (&data[0]) + i * dimCell;
    }

    inline dataType* operator[]( const int i)
    {
      return (&data[0]) + i * dimCell;
    }


    /* ---------- Operations ---------- */
    inline void MeanFilter( int wndRadius = 1 )
    {
      // Row Integaral
      dataType *tmp = new dataType[rows * cols * dimCell];
      for ( int i=0; i<rows; i++ ) {
        dataType *fastp = tmp + i * cols * dimCell;
        dataType *slowp = fastp;
        dataType *dp = &data[0] + i * cols * dimCell;
        for ( int c=0; c<dimCell; c++ ) {
          *(fastp++) = *(dp++);
        }
        for ( int j=1; j<cols; j++ ) {
          for ( int c=0; c<dimCell; c++ ) {
            *(fastp++) = *(dp++) + *(slowp++);
          }
        }
      }

      // Column Integral
      for ( int j=0; j<cols; j++ ) {
        dataType *slowp = tmp + j * dimCell;
        dataType *fastp = tmp + ( j + cols ) * dimCell;

        for ( int i=1; i<rows; i++ ) {
          for ( int c=0; c<dimCell; c++ ) {
            *(fastp++) += *(slowp++);
          }
          fastp += (cols-1) * dimCell;
          slowp += (cols-1) * dimCell;
        }
      }
      
      // mean calculation
      for ( int i=0; i<rows; i++ ) {
        for ( int j=0; j<cols; j++ ) {
          int top = i - wndRadius - 1;
          if ( top < 0 ) top = -1;
          int left = j - wndRadius - 1;
          if ( left < 0 ) left = -1;
          int bottom = i + wndRadius;
          if ( bottom >= rows ) bottom = rows - 1;
          int right = j + wndRadius;
          if ( right >= cols ) right = cols - 1;

          for ( int c=0; c<dimCell; c++ ) {
            dataType sum = tmp[ ( bottom * cols + right ) * dimCell + c ];
            if ( left >  -1 ) sum -= tmp[ ( bottom * cols + left ) * dimCell + c ];
            if ( top > -1 ) sum -= tmp[ ( top * cols + right ) * dimCell + c ];
            if ( left > -1 && top > -1 ) sum += tmp[ ( top * cols + left ) * dimCell + c ];
            int area = ( bottom - top ) * ( right - left );
            
            data[ ( i * cols + j ) * dimCell + c ] = sum / area;
          }
        }
      }
      
      DeleteToNullWithTestArray( tmp );
      
    }

    inline FeatImage<dataType> Group( int wndRadius )
    {
      int side = ((wndRadius << 1) + 1);
      int area = side * side * dimCell;

      FeatImage<dataType> img( rows, cols, area );
      for ( int i=0; i<rows; i++ ) {
        for ( int j=0; j<cols; j++ ) {
          
          dataType *ptr = img[ i * cols + j ];
          memset( ptr, 0, sizeof(dataType) * area * dimCell );
          for ( int dy=-wndRadius; dy<=wndRadius; dy++ ) {
            int y = i + dy;
            for ( int dx=-wndRadius; dx<=wndRadius; dx++ ) {
              int x = j + dx;
              if ( 0 <= y && y < rows && 0 <= x && x < cols ) {
                memcpy( ptr, (*this)(y,x), sizeof(dataType) * dimCell );
              }
              ptr += dimCell;
            }
          }

        }
      }

      return img;
    }

    
    inline void NormalizeCell()
    {
      for ( int i=0, end=rows*cols; i<end; i++ ) {
        normalize_vec<dataType>( (*this)[i], (*this)[i], dimCell );
      }
    }
    
    
    /* ---------- Operators ---------- */
    inline bool operator==( const FeatImage<dataType> &other )
    {
      if ( rows != other.rows || cols != other.cols || dimCell != other.dimCell ) {
        return false;
      }

      for ( int i=0; i<rows * cols * dimCell; i++ ) {
        if ( abs( data[i] - other.data[i] ) > 1e-5 ) {
          return false;
        }
      }

      return true;
    }

    /* ---------- Tracer ---------- */
    inline void showCell( const int i ) const
    {
      const dataType* f = (*this)(i);
      std::cout << "( ";
      for ( int c=0; c<dimCell; c++ ) {
        std::cout << f[c] << " ";
      }
      std::cout << ")\n";
    }

    inline void showCell( const int i, const int j ) const
    {
      const dataType* f = (*this)(i,j);
      std::cout.precision(5);
      std::cout << "( ";
      for ( int c=0; c<dimCell; c++ ) {
        std::cout << f[c] << " ";
      }
      std::cout << ")\n";
    }

  };
  
}
