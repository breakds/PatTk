#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "../FeatImage.hpp"
#include "../../interfaces/opencv_aux.hpp"



using namespace PatTk;
int main()
{

  auto img = cvFeat<HOG>::gen( "camvid_mid/Seq05VD_f00810.png" );

  
  float feat[img.GetPatchDim()];
  
  timer::tic();
  for ( int i=0; i<img.rows; i++ ) {
    for ( int j=0; j<img.cols; j++ ) {
      img.FetchPatch( i, j, feat );
      FeatImage<float>::PatchProxy a = img.Spawn( i, j );
      for ( int c=0; c<img.GetPatchDim(); c++ ) {
        // printf( "%.3f %.3f\n", feat[c], a(c) );
        int x = a(c);
      }

      char ch;
      scanf( "%c", &ch );
    }
  }
  printf( "time elapsed: %.4lf sec\n", timer::utoc() );
        
  return 0;
}
