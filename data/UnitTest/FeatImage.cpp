#include "LLPack/utils/extio.hpp"
#include "../FeatImage.hpp"
#include "../../interfaces/opencv_aux.hpp"


using namespace PatTk;
int main()
{

  auto img = cvFeat<HOG>::gen( "/scratch.1/breakds/data/camvid_mid/Seq05VD_f00810.png" );
  cvFeat<HOG>::options.orientation_bins = 12;
  Info( "%d\n", cvFeat<HOG>::options.orientation_bins );
  return 0;
}
