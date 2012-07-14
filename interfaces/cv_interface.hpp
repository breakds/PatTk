/*********************************************************************************
 * File: cv_interface.hpp
 * Description: PatTk interfaces with open CV library
 * by BreakDS, University of Wisconsin Madison, Fri Jul 13 13:14:44 CDT 2012
 *********************************************************************************/


#pragma once

#include "data/2d.hpp"
#include "data/features.hpp"
#include "opencv2/opencv.hpp"


namespace PatTk
{


  // +-------------------------------------------------------------------------------
  // |  Image<cellType, typename valueType> Generators
  // +-------------------------------------------------------------------------------
  
  template <typename cellType, typename valueType>
  class cvFeatGen
  {
  public:
    static Image<cellType,valueType> gen( const cv::Mat __attribute__((__unused__)) &raw )
    {
      return Image<cellType,valueType>();
    }
  };
  
  
  /// LabCell Specialization
  template <typename valueType>
  class cvFeatGen<LabCell, valueType>
  {
  public:
    static Image<LabCell, valueType> gen( const cv::Mat& raw )
    {
      Image<LabCell, valueType > img( raw.rows, raw.cols );
      cv::Mat lab;
      cvtColor( raw, lab, CV_BGR2Lab );

      int i = 0;
      for ( auto it = lab.begin<cv::Vec3b>(), end = lab.end<cv::Vec3b>(); it != end; it ++ ) {
        img.setCell( i, LabCell( (*it)[0], (*it)[1], (*it)[2] ) );
        i++;
      }
      return img;
    }
  };
  
};
