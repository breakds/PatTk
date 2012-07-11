/*********************************************************************************
 * File: features.hpp
 * Description: implementation of several commonly used feature cells
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:15:22 CDT 2012
 *********************************************************************************/

#pragma once

#include "2d.hpp"

namespace PatTk
{

  // Container for an CIE L*a*b* cell (raw Lab color feature)
  class LabCell : public AbstractCell<unsigned char>
  {
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

    inline unsigned char& operator[]( const int index )
    {
      return lab[index];
    }

    inline const unsigned char& operator()( const int index ) const
    {
      return lab[index];
    }
  };

  
};
