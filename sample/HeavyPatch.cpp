/*********************************************************************************
 * File: usage.cpp
 * Description: usage of 2d library.
 * by BreakDS, University of Wisconsin Madison, Fri Jul 13 23:12:59 CDT 2012
 *********************************************************************************/

#include <utility>
#include "data/FeatImage.hpp"
#include "interfaces/opencv_aux.hpp"
#include "LLPack/utils/extio.hpp"

using namespace PatTk;
using std::pair;


int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Not enough arguments." );
    exit( -1 );
  }


  auto img = cvFeat<HOG>::gen( argv[1], 2, 0.5 );
  
  
  float feat[img.GetPatchDim()];
  for ( int i=0; i<img.rows; i++ ) {
    for ( int j=0; j<img.cols; j++ ) {
      if ( img.FetchPatch( i, j, 0.0, 0.01, feat ) ) {
        DebugInfo( "(%d,%d): true", i, j );
      } else {
        DebugInfo( "(%d,%d): false", i, j );
      }
    }
    ResumeOnRet();
  }



  return 0;
}

