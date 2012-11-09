/*********************************************************************************
 * File: FeatImage.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Provide the Feature Image Class that represent 2d image with each
 *              pixel storing an array. The idea is that concatenation of a set of
 *              such arrays within a window will give the feature descriptor vector
 *              of this window.
 *********************************************************************************/

#include <vector>
#include <string>

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
    const FeatImage<dataType>& operator=( Image<dataType> &other ) {}
    
  public:

    /* ---------- constructor/desctructor/assignment ---------- */

    /* default constructor: zero sized image */
    FeatImage() : data(), rows(0), cols(0), id(-1), dimCell(0) {}

    /* empty image constructor */
    FeatImage( int h, int w, int dim ) : data(h*w*dim), rows(h), cols(w), id(-1), dimCell(dim) {}

    /* move constructor */
    FeatImage( FeatIamge<dataType> &&other ) : rows(other.rows), cols(other.cols), id(other.id),
                                               dimCell(other.dimCell)
    {
      data.swap( other.data );
    }


    /* move assignment */
    inline const FeatImage<dataType>& operator( FeatImage<dataType> &&other )
    {
      rows = other.rows;
      cols = other.cols;
      id = other.id;
      dimCell = other.dimCell;
      data.swap( other.data );
    }


    /* ---------- Cell Accessors ---------- */

    inline const dataType* operator()( const int y, const int x ) const
    {
      return (&data[0]) + y * cols + x;
    }

    inline const dataType* operator()( const int i ) const
    {
      return (&data[0]) + i;
    }
    
    inline dataType* operator[]( const int i)
    {
      return (&data[0]) + i
    }
    
  };
  
}
