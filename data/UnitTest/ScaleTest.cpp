#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../FeatImage.hpp"
#include "../../interfaces/opencv_aux.hpp"
#include "../../query/tree.hpp"



using namespace PatTk;
int main()
{

  srand(0);

  auto src = cvFeat<HOG>::gen( "simple2/src.png" , 4, 0.95 );
  src.SetRotBins( 9 );
  auto ref = cvFeat<HOG>::gen( "simple2/ref.png" );

  cv::Mat srcmat = cv::imread( "simple2/src.png" );
  ImageViewer srcv( "source", srcmat );
  cv::Mat refmat = cv::imread( "simple2/ref.png" );
  ImageViewer refv( "reference", refmat );

  float scale = static_cast<float>( src.cols ) / ref.cols;

  printf( "%.4f\n", scale );


  refv.setCallback( [&refmat,&srcv,&src,&ref,&scale]( int x, int y )
                    {
                      std::vector<PatLoc> list;
                      
                      
                      float y1 = y * scale;
                      float x1 = x * scale;

                      list.push_back( PatLoc( -1, y1, x1, 0.0, scale ) );
                      

                      srcv.display( list );

                      


                      float feat_ref[ref.GetPatchDim()];
                      float feat_src[src.GetPatchDim()];

                      ref.FetchPatch( y, x, feat_ref );
                      src.FetchPatch( y1, x1, 0.0, scale, feat_src );


                      printf( "------------------------------\n" );
                      for ( int i=0; i<ref.GetPatchDim(); i++ ) {
                        if ( 0 == i%9 ) {
                          printf( "--\n" );
                        }
                        printf( "%.4f\t%.4f\n", feat_ref[i], feat_src[i] );

                      }
                      
                      printf( "norm0: %.4f\n", norm_l2( feat_ref, ref.GetPatchDim() ) );
                      printf( "norm1: %.4f\n", norm_l2( feat_src, src.GetPatchDim() ) );
                      printf( "(%d,%d)\n", y, x );
                      
                      
                    } );
  

  while ( 27 != cv::waitKey(30) );

  return 0;
}
