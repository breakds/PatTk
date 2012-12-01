#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/tree.hpp"
#include "../graph/Trans.hpp"

using namespace PatTk;



int main()
{

  srand( 345645631 );


  cv::Mat refimg = cv::imread( "simple/ref.png" );
  
  
  auto img = cvFeat<HOG>::gen( "simple/ref.png", 8, 0.9 );
  auto ref = cvFeat<HOG>::gen( "simple/ref.png", 8, 0.9 );


  float feat[img.GetPatchDim()];

  cv::Mat srcmat = cv::imread( "simple/ref.png" );
  ImageViewer srcv( "source", srcmat );
  cv::Mat refmat = cv::imread( "simple/ref.png" );
  ImageViewer refv( "reference", refmat );


  Info( "Loading Tree ..." );
  auto tree = Tree<SimpleKernel<float> >::read( "tree.dat" );
  Done( "Loaded" );
  Info( "Querying ..." );

  IconList icons( "test", 13 );

  


  srcv.setCallback( [&img,&ref,&feat,&refv,&tree,&refimg,&icons]( int x, int y )
                    {

                      img.FetchPatch( y, x, feat );

                      printf( "query (%d,%d)\n", y, x );
                      
                      const std::vector<LocInfo>& re = tree->query( feat );
                      
                      printf( "size: %ld\n", re.size() );

                      int i = 0;
                      icons.clear();
                      
                      for ( auto& ele : re ) {
                        if ( i++ > 100 ) break;
                        icons.push( refimg, PatLoc( ele ) );
                        printf( "%d,%d,%d\n", ele.id, ele.y, ele.x );
                      }

                      icons.display();
                      
                    } );

  while( 27 != cv::waitKey() ) ;

  return 0;
}
