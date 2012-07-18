/*********************************************************************************
 * File: TreeBuild.cpp
 * Description: Demonstrate the usage of tree library
 * by BreakDS, University of Wisconsin Madison, Sat Jul 14 09:41:29 CDT 2012
 *********************************************************************************/
#include <type_traits>
#include "LLPack/utils/extio.hpp"
#include "LLPack/algorithms/random.hpp"
#include "data/2d.hpp"
#include "data/features.hpp"
#include "query/tree.hpp"
#include "query/kernel.hpp"
#include "opencv2/opencv.hpp"
#include "interfaces/cv_interface.hpp"

#define STRIDE 5
#define CELL_STRIDE 1
#define PATCH_SIZE 10

using namespace PatTk;

int main( int argc, char **argv )
{

  srand( 1342464046 );


  if ( argc < 2 ) {
    Error( "Not enough arguments." );
    exit( -1 );
  }


  cv::Mat mat = cv::imread( argv[1] );

  // Create an image with CIEL*a*b* feature descriptors
  auto img = cvFeatGen<LabCell,int>::gen( mat );

  // Set the original image path for img
  // This is going to be used in visualization
  img.setFullPath( argv[1] );
  
  // Set the parameter for patches
  // in this case, will be 5 x 5 patch with a cell stride of 4
  img.SetPatchParameter( PATCH_SIZE, PATCH_SIZE, CELL_STRIDE );

  // Create vector of patches
  vector<Image<LabCell,int>::Patch> patches;
  for ( int y=0; y<img.rows; y+=STRIDE ) {
    for ( int x=0; x<img.cols; x+=STRIDE ) {
      auto patch = img.Spawn( y, x );
      if ( patch.isValid() ) {
        patches.push_back( std::move(patch) );
      }
    }
  }
  
  printf( "in total: %ld\n", patches.size() );

  // Create the tree
  Tree<BasicKernel<LabCell,int> > tree( patches );

  // Tree Query
  const Tree<BasicKernel<LabCell,int> > *node = tree.direct( img.Spawn( 0, 0 ) );

  // Show the node
  node->Summary();
  

  return 0;


}
