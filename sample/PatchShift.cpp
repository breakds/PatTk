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
  Concat<LabCell, LabCell> a( LabCell( 12, 12 ,12 ), LabCell( 255, 255 ,255 ));
  a.Summary();
  return 0;
}

