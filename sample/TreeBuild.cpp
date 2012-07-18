/*********************************************************************************
 * File: TreeBuild.cpp
 * Description: Demonstrate the usage of tree library
 * by BreakDS, University of Wisconsin Madison, Sat Jul 14 09:41:29 CDT 2012
 *********************************************************************************/
#include <type_traits>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/utils/Environment.hpp"
#include "data/2d.hpp"
#include "data/features.hpp"
#include "query/tree.hpp"
#include "query/kernel.hpp"
#include "opencv2/opencv.hpp"
#include "interfaces/cv_interface.hpp"


using namespace EnvironmentVariable;
using namespace PatTk;

int main( int argc, char **argv )
{
  
  srand( 1342464046 );
  

  if ( argc < 2 ) {
    Error( "Not enough arguments." );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  // Create an album with CIEL*a*b* feature descriptors
  auto album = cvAlbumGen<LabCell,int>::gen( path::FFFL( env["folder"], env["files"], ".png" ) );
  album.SetPatchParameter( env["patch-size"], env["patch-size"], env["cell-stride"] );

  // Create vector of patches
  vector<Image<LabCell,int>::Patch> patches;
  for ( int i=0; i<album.size(); i++ ) {
    const auto& img = album(i);
    for ( int y=0; y<img.rows; y+=env["patch-stride"] ) {
      for ( int x=0; x<img.cols; x+=env["patch-stride"] ) {
        auto patch = img.Spawn( y, x );
        if ( patch.isValid() ) {
          patches.push_back( std::move(patch) );
        }
      }
    }
  }
  
  printf( "in total: %ld\n", patches.size() );
  
  // Create the tree
  Tree<BasicKernel<LabCell,int> > tree( patches );

  // Tree Query
  const Tree<BasicKernel<LabCell,int> > *node = tree.direct( album(0).Spawn( 0, 0 ) );

  // Show the node
  node->Summary();
  

  return 0;


}


// Appendix: A sample configuration file
// ;; Files
// (folder camvid_mid)
// (files Seq05VD_f00810
//        Seq05VD_f00840)


// ;; Patch
// (patch-stride 5)
// (cell-stride 1)
// (patch-size 10)


// ;; Random Tree
// (proj-dim 8)   
// (hypo-num 20)
// (converge-th 120)
// (set-size-a 50)
// (set-size-b 1000)
