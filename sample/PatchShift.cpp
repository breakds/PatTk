/*********************************************************************************
 * File: PatchShift.cpp
 * Description: Sample program for using 2d library. Search for the nearest neighbor
 *              in image A from image B. Search is limited within a range.
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:45:21 CDT 2012
 *********************************************************************************/

#include "data/features.hpp"

using namespace PatTk;

int main()
{

  HistCell<int> b(vector<int>({1,2,3}));
  b.Summary();
  Concat<unsigned char, LabCell, HistCell<unsigned char>> a( LabCell( 12, 12 ,12 ),
                                                   HistCell<unsigned char>( vector<unsigned char>( {255, 255 ,255} ) ) );
  a.Summary();
  return 0;
}

