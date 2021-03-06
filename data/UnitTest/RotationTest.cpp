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

  float ang = 30.0 / 180.0 * M_PI;

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
                      
                      float y1 = cosa * y + sina * x;
                      float x1 = -sina * y + cosa * x + refmat.rows * sina;





                      
                      list.push_back( PatLoc( -1, y1, x1, ang, 1.0 ) );

                      srcv.display( list );

                      


                      float feat_ref[ref.GetPatchDim()];
                      float feat_src[src.GetPatchDim()];

                      ref.FetchPatch( y, x, feat_ref );
                      src.FetchPatch( 2, y1, x1, ang, 1.0, feat_src );



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
