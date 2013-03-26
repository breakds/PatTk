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


  auto img = cvFeat<HOG>::gen( argv[1], 2, 0.8 );

  float feat[img.GetPatchDim()];

  if ( img.FetchPatch( 1, 170, 1.25, 0.0, 0.8, feat ) ) {
    DebugInfo( "true" );
  } else {
    DebugInfo( "false" );
  }
  

  // for ( int i=0; i<img.rows; i++ ) {
  //   for ( int j=0; j<img.cols; j++ ) {
  //     if ( img.FetchPatch( 3, i, j, 0.0, feat ) ) {
  //       DebugInfo( "(%d,%d): true", i, j );
  //     } else {
  //       DebugInfo( "(%d,%d): false", i, j );
  //     }
  //   }
  //   ResumeOnRet();
  // }
  


  return 0;
}

