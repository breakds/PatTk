/*********************************************************************************
 * File: opencv_aux.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Provide auxiliary routines as interfaces between opencv and PatTk
 *********************************************************************************/


#pragma once
#include <string>
#include <type_traits>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/candy.hpp"
#include "LLPack/utils/time.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"

#define _USE_MATH_DEFINES

namespace PatTk
{

  /* anonymous namespace for helper functions */
  namespace
  {

  }

  enum featEnum { BGR, Lab, HOG, DEFAULT_FEAT };

  
  template <featEnum featType> struct FeatOptions {};

  
  template<>
  struct FeatOptions<HOG>
  {
    int orientation_bins;
    int cell_side;
    int block_size;
    bool enable_color;
    bool gaussian_filter;
    FeatOptions() : orientation_bins(9), cell_side(5), block_size(3),
                    enable_color(true), gaussian_filter(false)
    {}

  };


  
  
  


  template <featEnum featType=DEFAULT_FEAT>
  class cvFeat
  {
  public:

    static FeatOptions<featType> options;
    
    template <featEnum T=featType>
    static FeatImage<unsigned char> gen( std::string filename, ENABLE_IF(DEFAULT_FEAT==T) )
    {
      return FeatImage<unsigned char>();
    }
    
    // +--------------------------------------------------+
    // |  BGR Feature Generator                           |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<uchar> gen( std::string filename, ENABLE_IF( BGR == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<BGR>::gen()   Failed to load image %s", filename.c_str() );
        exit( -1 );
      }
      
      if ( 1 == raw.channels() ) {
        FeatImage<uchar> img( raw.rows, raw.cols, 3 );
        uchar *img_ptr = img[0];
        for ( int i=0; i<raw.rows; i++ ) {
          uchar *raw_ptr = raw.ptr<uchar>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            *(img_ptr++) = *raw_ptr;
            *(img_ptr++) = *raw_ptr;
            *(img_ptr++) = *raw_ptr;
            raw_ptr++;
          }
        }
        return img;
      } else if ( 3 == raw.channels() ) {
        FeatImage<uchar> img( raw.rows, raw.cols, 3 );
        uchar *img_ptr = img[0];
        size_t row_size = sizeof(uchar) * 3 * raw.cols;
        int row_stride = 3 * raw.cols;
        for ( int i=0; i<raw.rows; i++ ) {
          uchar *raw_ptr = raw.ptr<uchar>(i);
          memcpy( img_ptr, raw_ptr, row_size );
          img_ptr += row_stride;
        }
        return img;
      }

      Error( "cvFeat<BGR>::gen()   Bad number of channels: %d\n", raw.channels() );
      exit( -1 );
    }


    // +--------------------------------------------------+
    // |  Lab Feature Generator                           |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<uchar> gen( std::string filename, ENABLE_IF( Lab == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<Lab>::gen()   Failed to load image %s", filename.c_str() );
        exit( -1 );
      }
      
      if ( 3 == raw.channels() ) {
        cv::Mat lab;
        cv::cvtColor( raw, lab, CV_BGR2Lab );
        FeatImage<uchar> img( lab.rows, lab.cols, 3 );
        uchar *img_ptr = img[0];
        size_t row_size = sizeof(uchar) * 3 * lab.cols;
        int row_stride = 3 * lab.cols;
        for ( int i=0; i<lab.rows; i++ ) {
          uchar *lab_ptr = lab.ptr<uchar>(i);
          memcpy( img_ptr, lab_ptr, row_size );
          img_ptr += row_stride;
        }
        return img;
      }

      Error( "cvFeat<Lab>::gen()   Bad number of channels: %d\n", raw.channels() );
      exit( -1 );
    }



    template <featEnum T=featType>
    inline static void Angle2Bin( float angle, int &lo_bin, int &hi_bin, float &lo_wt, float &hi_wt,
                                  ENABLE_IF( HOG == T ) )
    {
      /* centers are 0, span, 2*span, 3*span, ..., (bins-1) * span,
       * where bins * span and 0 are coincident centers. each center
       * spans +-(span/2), and span (bin_span) is defined right below.
       */
      static float bin_span = M_PI / options.orientation_bins;
      float tmp = angle < 0.0 ? M_PI + angle : angle;

      lo_bin = tmp / bin_span;
      hi_bin = lo_bin + 1;
      if ( options.orientation_bins == hi_bin ) hi_bin = 0;
      hi_wt = tmp / bin_span - lo_bin;
      lo_wt = 1.0 - hi_wt;
    }
      


    // +--------------------------------------------------+
    // |  HOG Feature Generator                           |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<float> gen( std::string filename, ENABLE_IF( HOG == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<HOG>::gen()   Failed to load image %s", filename.c_str() );
        exit( -1 );
      }
      FeatImage<float> img;


      
      /* Image Normalization */
      cv::Mat normalized;
      cv::normalize( raw, normalized, 0, 255, cv::NORM_MINMAX );


      /* gradient */
      cv::Mat gradx( raw.rows, raw.cols, CV_32FC1 );
      cv::Mat grady( raw.rows, raw.cols, CV_32FC1 );
      if ( options.enable_color && 3 == normalized.channels() ) {
        cv::Mat multiGradx, multiGrady;
        cv::Sobel( normalized, multiGradx, CV_32F, 1, 0 );
        cv::Sobel( normalized, multiGrady, CV_32F, 0, 1 );

        for ( int i=0; i<raw.rows; i++ ) {
          float *mxptr = multiGradx.ptr<float>(i);
          float *myptr = multiGrady.ptr<float>(i);
          float *gxptr = gradx.ptr<float>(i);
          float *gyptr = grady.ptr<float>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            float max = (*mxptr) * (*mxptr) + (*myptr) * (*myptr);
            mxptr++;
            myptr++;
            for ( int k=1; k<3; k++ ) {
              float norm = (*mxptr) * (*mxptr) + (*myptr) * (*myptr);
              if ( norm > max ) {
                max = norm;
                *gxptr = *mxptr;
                *gyptr = *myptr;
              }
              mxptr++;
              myptr++;
            }
            gxptr++;
            gyptr++;
          }
        }
      } else if ( 1 == normalized.channels() ) {
        cv::Sobel( normalized, gradx, CV_32F, 1, 0 );
        cv::Sobel( normalized, grady, CV_32F, 0, 1 );
      }

      /* convert to rad and norm */
      cv::Mat radMat( raw.rows, raw.cols, CV_32FC1 );
      cv::Mat normMat( raw.rows, raw.cols, CV_32FC1 );

      for ( int i=0; i<raw.rows; i++ ) {
        float *gxptr = gradx.ptr<float>(i);
        float *gyptr = grady.ptr<float>(i);
        float *rad_ptr = radMat.ptr<float>(i);
        float *norm_ptr = normMat.ptr<float>(i);
        for ( int j=0; j<raw.cols; j++ ) {
          *(rad_ptr++) = atan2( *gyptr, *gxptr );
          *(norm_ptr++) = sqrt( (*gyptr) * (*gyptr) + (*gxptr) * (*gxptr) );
          gyptr++;
          gxptr++;
        }
      }

      /* initialization of voting */
      FeatImage<float> pointMap( raw.rows, raw.cols, options.orientation_bins );
      int lo_bin = 0;
      int hi_bin = 0;
      float lo_wt = 0.0f;
      float hi_wt = 0.0f;
      {
        int k = 0;
        for ( int i=0; i<raw.rows; i++ ) {
          float *rad_ptr = radMat.ptr<float>(i);
          float *norm_ptr = normMat.ptr<float>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            Angle2Bin( *rad_ptr, lo_bin, hi_bin, lo_wt, hi_wt );
            pointMap[k][lo_bin] += lo_wt * (*norm_ptr);
            pointMap[k][hi_bin] += hi_wt * (*norm_ptr);
            rad_ptr++;
            norm_ptr++;
            k++;
          }
        }
      }
      
      pointMap.MeanFilter( options.cell_side >> 1 );

      FeatImage<float> blockMap = pointMap.Group( options.block_size >> 1 );

      blockMap.NormalizeCell();
      
      return blockMap;
    }

  };
}

/* definitions for static members */
template <PatTk::featEnum featType>
PatTk::FeatOptions<featType> PatTk::cvFeat<featType>::options;




