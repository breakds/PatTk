/*********************************************************************************
 * File: PatchShift.cpp
 * Description: Sample program for using 2d library. Search for the nearest neighbor
 *              in image A from image B. Search is limited within a range.
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:45:21 CDT 2012
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
  Image<LabCell,std::pair<int,int> > img = cvFeatGen<LabCell,std::pair<int,int> >::gen( mat );

  // Set the parameter for patches
  // in this case, will be 5 x 5 patch with a cell stride of 4
  img.SetPatchParameter( 5, 5, 4 );


  // Get a patch at coordinate (35, 102)
  Image<LabCell, std::pair<int,int> >::Patch patch = img.Spawn( 35, 102 );


  // Iterate the cell components of this patch, and print them out
  for ( int i=0; i<patch.dim(); i++ ) {
    patch(i).Summary();
  }

  // Iterate the cell components of the patch
  for ( int i=0; i<patch.dim(); i++ ) {
    // Iterate the feature descriptors component of the cell
    for ( int j=0; j<patch(i).length; j++ ) {
      printf( "%hhu ", patch(i)(j) );
    }
  }
  printf( "\n" );
  
  
  return 0;
}

