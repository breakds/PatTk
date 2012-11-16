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
    }
  }
  printf( "time elapsed: %.4lf sec\n", timer::utoc() );
        
  return 0;
}
