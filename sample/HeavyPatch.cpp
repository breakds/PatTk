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

  {
    cv::Mat mat = cv::imread( argv[1] );

    // Create an image with CIEL*a*b* feature descriptors
    auto img = cvFeatGen<LabCell,std::pair<int,int>,false >::gen( mat );

    // Set the parameter for patches
    // in this case, will be 5 x 5 patch with a cell stride of 4
    img.SetPatchParameter( 3, 3, 1 );

    auto patch = img.Spawn( 3, 24, 1, 0 );

    // Iterate the patch components directly
    for ( int i=0; i<patch.dim(); i++ ) {
      printf( "%3hhu ", patch[i] );
    }
    printf( "\n" );
  }

  printf( "------------------------------\n" );
  {
    cv::Mat mat = cv::imread( argv[2] );

    // Create an image with CIEL*a*b* feature descriptors
    auto img = cvFeatGen<LabCell,std::pair<int,int>,false>::gen( mat );

    // Set the parameter for patches
    // in this case, will be 5 x 5 patch with a cell stride of 4
    img.SetPatchParameter( 3, 3, 1 );

    img( 18,165 ).Summary();
    img( 18,166 ).Summary();
    img( 19,166 ).Summary();
    img( 19,165 ).Summary();
    
    auto patch = img.Spawn( 18.77103, 165.85076, 1, 45 );
    // Iterate the patch components directly
    for ( int i=0; i<patch.dim(); i++ ) {
      printf( "%3hhu ", patch[i] );
    }
    printf( "\n" );


  }


  


  return 0;
}

