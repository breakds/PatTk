/*********************************************************************************
 * File: IconList.hpp
 * Description: Icon List Test for rotated patches (Unit Test for IconList)
 * by BreakDS, University of Wisconsin Madison, Mon Aug 27 13:36:50 CDT 2012
 *********************************************************************************/

#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "../opencv_aux.hpp"
#include <string>
#include <vector>
#include <cmath>

using namespace PatTk;




int main()
{

  IconList icons( "test", 13 );

  std::vector<std::string> files;

  files.push_back( "simple/ref.png" );

  
  icons.push( files, PatLoc( 0, 200, 45, 0.0, 1.0 ) );

  icons.push( files, PatLoc( 0, 100, 100, 0.0, 1.0 ) );

  icons.display();

  cv::waitKey();
  
  return 0;
}
