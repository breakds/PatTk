/*********************************************************************************
 * File: usage.cpp
 * Description: usage of 2d library.
 * by BreakDS, University of Wisconsin Madison, Fri Jul 13 23:12:59 CDT 2012
 *********************************************************************************/

#include <utility>
#include "data/features.hpp"
#include "opencv2/opencv.hpp"
#include "interfaces/cv_interface.hpp"

using namespace PatTk;
using std::pair;


int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Not enough arguments." );
    exit( -1 );
  }
  cv::Mat mat = cv::imread( argv[1] );

  // Create an image with CIEL*a*b* feature descriptors
  auto img = cvFeatGen<LabCell,std::pair<int,int>,false>::gen( mat );

  // Set the parameter for patches
  // in this case, will be 5 x 5 patch with a cell stride of 4
  img.SetPatchParameter( 5, 5, 4 );


  // Get a patch at coordinate (35, 102)
  auto patch = img.Spawn( 54, 266, 1.8465, 18.8965 );

  // Iterate the patch components directly
  for ( int i=0; i<patch.dim(); i++ ) {
    printf( "%hhu ", patch[i] );
  }
  printf( "\n" );
  

  


  return 0;
}

