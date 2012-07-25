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


#define CELL HoGCell

using namespace EnvironmentVariable;
using namespace PatTk;


struct MCP
{
  cv::Mat& testImg;
  Image<CELL,int>& target;
  IconList<Image<CELL,int>::Patch>& dbg;
  Tree<BasicKernel<CELL,int> >& tree;
};


void trace( const Image<HoGCell,int>::Patch& p )
{
  printf( "+--------------------------------------------------\n" );
  for ( int i=0; i<p.cellNum(); i++ ) {
    printf( "| " );
    int sum = 0;
    for ( int j=0; j<p(i).length; j++ ) {
      printf( "%3hhu ", p(i)(j) );
      sum += p(i)(j);
    }
    printf( "sum = %d\n", sum );
  }
  printf( "+--------------------------------------------------\n\n" );
}


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
  auto album = cvAlbumGen<CELL,int>::gen( path::FFFL( env["folder"], env["files"], ".png" ) );
  album.SetPatchParameter( env["patch-size"], env["patch-size"], env["cell-stride"] );

  
  // Create vector of patches
  vector<Image<CELL,int>::Patch> patches;
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
  Tree<BasicKernel<CELL,int> > tree( patches );

  // Write the tree out
  tree.write( strf( "%s/%s.dat", env["folder"].c_str(), env["tree-prefix"].c_str() ) );
  Info( "Tree Written to %s/%s.dat.", env["folder"].c_str(), env["tree-prefix"].c_str() );
  

  // Tree Query
  const Tree<BasicKernel<CELL,int> > *node = tree.direct( album(0).Spawn( 0, 0 ) );

  // Show the node
  node->Summary();
  
  cv::Mat testImg = cv::imread( strf( "%s/%s.png", env["folder"].c_str(), env["target"].c_str() ) );

  
  auto target = cvFeatGen<CELL,int>::gen( testImg );
  target.SetPatchParameter( env["patch-size"], env["patch-size"], env["cell-stride"] );

  cv::imshow( "show", testImg );


  IconList<Image<CELL,int>::Patch> dbg( "debug" );
  MCP mcp = { testImg, target, dbg, tree };

  
  cv::setMouseCallback( "show",
                        []( int event, int x, int y,
                            int __attribute__((__unused__)) flags, void *param )
                        {
                          if ( CV_EVENT_LBUTTONDOWN == event) {
                            MCP *mcp = (MCP*) param;

                            int width = mcp->target.GetPatchWidth();
                            int height = mcp->target.GetPatchHeight();

                            int cx = x - ( width >> 1 );
                            int cy = y - ( height >> 1 );


                            mcp->target.Summary();
                            if ( 0 <= cx && cx + width <= mcp->testImg.cols &&
                                 0 <= cy && cy + height <= mcp->testImg.rows ) {
                              cv::Mat canvas = mcp->testImg.clone();
                              rectangle( canvas,
                                         cv::Point( cx, cy ),
                                         cv::Point( cx + width,
                                                    cy + height ),
                                         cv::Scalar( 0, 255, 0 ) ) ;
                              cv::imshow( "show", canvas );

                              const Tree<BasicKernel<CELL,int> > *node =
                                mcp->tree.direct( mcp->target.Spawn( cy, cx ) );
                              mcp->dbg.clear();
                              mcp->dbg.options.zoom = 2;
                              for ( auto& patch : node->patches ) {
                                mcp->dbg.push( patch );
                              }
                              mcp->dbg.display();
                            }
                          }
                        }, &mcp );
  while ( 27 != cv::waitKey(30) ) ;
  
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
