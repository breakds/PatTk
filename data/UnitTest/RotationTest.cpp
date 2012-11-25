#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/list.hpp"
#include "../FeatImage.hpp"
#include "../../interfaces/opencv_aux.hpp"
#include "../../query/tree.hpp"



using namespace PatTk;
int main()
{

  auto src = cvFeat<HOG>::gen( "simple/src.png.30" , 2, 0.9 );
  auto ref = cvFeat<HOG>::gen( "simple/ref.png", 2, 0.9f );

  cv::Mat srcmat = cv::imread( "simple/src.png.30" );
  ImageViewer srcv( "source", srcmat );
  cv::Mat refmat = cv::imread( "simple/ref.png" );
  ImageViewer refv( "reference", refmat );


  float ang = 0.0;

  int N = 10;

  float feat_base[ref.GetPatchDim()];
  
  Circular<float*> features( N );

  int curY(0), curX(0);

  refv.setCallback( [&refmat,&srcv,&src,&ref,&ang,&feat_base,&features,&curY,&curX]( int x, int y )
                    {

                      curY = y;
                      curX = x;
                      
                      ref.FetchPatch( y, x, feat_base );

                      features.clear();

                      ang = 0.0;

                      float *feat = new float[ref.GetPatchDim()];

                      ref.FetchPatch( 2, y, x, ang, 1.0, feat );

                      features.push_back( feat );

                      printf( "features.size = %d\n", features.size() );
                      
                      printf( "------------------------------\n" );
                      for ( int i=0; i<ref.GetPatchDim(); i++ ) {
                        if ( 0 == i%9 ) {
                          printf( "--\n" );
                        }
                        printf( "%.4f   |", feat_base[i] );
                        for ( int k=0; k<features.size(); k++ ) {
                          printf( "\t%.4f", features(k)[i] );
                        }
                        printf( "\n" );
                      }

                      printf( "(%d,%d)\n", y, x );
                    } );

  char key;
  while ( 27 != ( key = cv::waitKey(30) ) ) {
    if ( 81 == key ) {
      std::vector<PatLoc> list;
      ang -= 1.0 / 180.0 * M_PI;
      list.push_back( PatLoc( -1, curY, curX, ang, 1.0f ) );
      refv.display( list );

      float *feat = new float[ref.GetPatchDim()];

      ref.FetchPatch( 2, curY, curX, ang, 1.0, feat );

      if ( features.full() ) {
        features.pop_front();
      }

      features.push_back( feat );

      printf( "------------------------------\n" );
      for ( int i=0; i<ref.GetPatchDim(); i++ ) {
        if ( 0 == i%9 ) {
          printf( "--\n" );
        }
        printf( "%.4f   |", feat_base[i] );
        for ( int k=0; k<features.size(); k++ ) {
          printf( "\t%.4f", features(k)[i] );
        }
        printf( "\n" );
      }

      
    } else if ( 83 == key ) {
      std::vector<PatLoc> list;
      ang += 1.0 / 180.0 * M_PI;
      list.push_back( PatLoc( -1, curY, curX, ang, 1.0f ) );
      refv.display( list );

      float *feat = new float[ref.GetPatchDim()];

      ref.FetchPatch( 2, curY, curX, ang, 1.0, feat );

      if ( features.full() ) {
        features.pop_front();
      }
      
      features.push_back( feat );

      printf( "------------------------------\n" );
      for ( int i=0; i<ref.GetPatchDim(); i++ ) {
        if ( 0 == i%9 ) {
          printf( "--\n" );
        }
        printf( "%.4f   |", feat_base[i] );
        for ( int k=0; k<features.size(); k++ ) {
          printf( "\t%.4f", features(k)[i] );
        }
        printf( "\n" );
      }

    }
  }


  return 0;
}
