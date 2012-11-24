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

  float ang = M_PI / 6;

  auto src = cvFeat<HOG>::gen( "simple/src.png.30" , 2, 0.9 );
  src.SetRotBins( 9 );
  auto ref = cvFeat<HOG>::gen( "simple/ref.png" );

  cv::Mat srcmat = cv::imread( "simple/src.png.30" );
  ImageViewer srcv( "source", srcmat );
  cv::Mat refmat = cv::imread( "simple/ref.png" );
  ImageViewer refv( "reference", refmat );

  refv.setCallback( [&refmat,&srcv,&src,&ref,&ang]( int x, int y )
                    {
                      std::vector<PatLoc> list;
                      

                      float cosa = cosf( ang );
                      float sina = sinf( ang );
                      
                      int y1 = static_cast<int>( cosa * y + sina * x );
                      int x1 = static_cast<int>( -sina * y + cosa * x + refmat.rows * sina );

                      // int y1 = 239 - y;
                      // int x1 = 319 - x;
                      
                      list.push_back( PatLoc( -1, y1, x1, ang, 1.0 ) );


                      float feat_ref[ref.GetPatchDim()];
                      float feat_src[src.GetPatchDim()];

                      ref.FetchPatch( y, x, feat_ref );
                      src.FetchPatch( 2, y1, x1, ang, 1.0, feat_src );

                      srcv.display( list );

                      printf( "------------------------------\n" );
                      for ( int i=0; i<ref.GetPatchDim(); i++ ) {
                        if ( 0 == i%9 ) {
                          printf( "--\n" );
                        }
                        printf( "%.4f\t%.4f\n", feat_ref[i], feat_src[i] );
                      }

                      printf( "norm0: %.4f\n", norm_l2( feat_ref, ref.GetPatchDim() ) );
                      printf( "norm1: %.4f\n", norm_l2( feat_src, src.GetPatchDim() ) );
                    } );
  

  while ( 27 != cv::waitKey(30) );

  return 0;
}
