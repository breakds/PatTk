/*********************************************************************************
 * File: FeatImage.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Provide the Feature Image Class that represent 2d image with each
 *              pixel storing an array. The idea is that concatenation of a set of
 *              such arrays within a window will give the feature descriptor vector
 *              of this window.
 *********************************************************************************/

#pragma once
#include <cassert>
#include <vector>
#include <iostream>
#include <string>
#include "LLPack/utils/SafeOP.hpp"
#include "vector.hpp"

namespace PatTk
{


  
  
  template <typename dataType>
  class FeatImage
  {
  private:
    /* ---------- data ---------- */
    std::vector<std::vector<dataType> > pyramid;
    std::vector<float> inv_norm;
    float scale_base;
    int scales;
    std::vector<float> scale_layer;
    std::vector<int> rows_layer;
    std::vector<int> cols_layer;
    
    

    /* ---------- Extra Properties ---------- */
    struct PatchOptions
    {
      int patch_size;
      int patch_stride;
      int patch_start_offset;
      int hist_dim;
      int patch_dim;
      int rotBins; // rotBins = 0 means no need to do rotation adjustment, otherwise do adjustment based this number of bins
      float shiftUnit; // unit for rotation adjustment
      std::vector<int> offset;
      std::vector<int> offsetY;
      std::vector<int> offsetX;
      bool normalized;
      PatchOptions( int dimCell ) : patch_size(5), patch_stride(4), 
                                    patch_start_offset( - ( patch_size >> 1 ) * patch_stride ),
                                    hist_dim( dimCell ),
                                    patch_dim( patch_size * patch_size * dimCell ),
                                    rotBins(0), shiftUnit(1.0),
                                    normalized(true) {}
    };

    /* ---------- prohibited methods ---------- */
    FeatImage( FeatImage<dataType>& other ) {}
    const FeatImage<dataType>& operator=( FeatImage<dataType> &other ) {}

  public:
    /* ---------- Simple Patch ---------- */
    
    class PatchProxy {
    public:
      const FeatImage<dataType> *parent;
      int y, x, coorIdx;
      float inv;

      
      /* constructor */
      PatchProxy( const FeatImage<dataType> *p, int y1, int x1 ) 
        : parent(p), y(y1), x(x1) 
      {
        coorIdx = (y1 * parent->cols + x1) * parent->dimCell;
        inv = parent->inv_norm[ y1 * parent->cols + x1 ];
      }

      /* get dimension */
      inline int dim() const
      {
        return parent->GetPatchDim();
      }

      inline int size() const
      {
        return parent->GetPatchDim();
      }
        
      
      inline dataType operator()( int c ) const
      {
        int y1 = parent->options.offsetY[c] + y;
        int x1 = parent->options.offsetX[c] + x;
        
        if ( y1 < 0 || y1 >= parent->rows ||
             x1 < 0 || x1 >= parent->cols ) {
          return 0;
        }
        
        return parent->get( coorIdx + parent->options.offset[c] ) * inv;
      }

      inline dataType operator[]( int c ) const
      {
        int y1 = parent->options.offsetY[c] + y;
        int x1 = parent->options.offsetX[c] + x;
        
        if ( y1 < 0 || y1 >= parent->rows ||
             x1 < 0 || x1 >= parent->cols ) {
          return 0;
        }

        return parent->get( coorIdx + parent->options.offset[c] ) * inv;
      }


      inline int id() const
      {
        return parent->id;
      }
    };
    
  public:
    /* ---------- properties ---------- */
    int rows, cols;
    int id;
    int dimCell;
    
  private:
    /* ---------- Patch Options ----------*/
    PatchOptions options;
    
  public:
    /* ---------- Patch Options Accessors ---------- */
    inline void SetPatchSize( int s ) 
    {
      options.patch_size = s;
      options.patch_start_offset = - ( s >> 1 ) * options.patch_stride;
      options.patch_dim = s * s * dimCell;
      initPatchOptions();
    }

    inline void SetPatchStride( int s )
    {
      options.patch_stride = s;
      options.patch_start_offset = - ( options.patch_size >> 1 ) * s;
      initPatchOptions();
    }

    inline int GetPatchDim() const
    {
      return options.patch_dim;
    }

    inline void SetRotBins( int r )
    {
      options.rotBins = r;
      if ( 0 != r ) {
        options.shiftUnit = M_PI / r;
      }
    }

    inline void ToggleNormalized( bool tog )
    {
      options.normalized = tog;
    }

    
  public:

    /* ---------- constructor/desctructor/assignment ---------- */

    /* initialize Patch Offsets Vector */
    inline void initPatchOptions()
    {
      options.offset.resize( options.patch_dim );
      options.offsetX.resize( options.patch_dim );
      options.offsetY.resize( options.patch_dim );
      int y = options.patch_start_offset;
      int k = 0;

      for ( int i=0; i<options.patch_size; i++, y+=options.patch_stride ) {
        int x = options.patch_start_offset;
        for ( int j=0; j<options.patch_size; j++, x+=options.patch_stride ) {
          for ( int c=0; c<dimCell; c++, k++ ) {
            options.offset[k] = ( y * cols + x ) * dimCell + c;
            options.offsetY[k] = y;
            options.offsetX[k] = x;
          }
        }
      }
    }


    void InitInvNorms()
    {
      inv_norm.resize( cols * rows );
      int k = 0;
      for ( int i=0; i<rows; i++ ) {
        for ( int j=0; j<cols; j++ ) {
          dataType sum = 0.0f;
          int y = i + options.patch_start_offset;
          for ( int sy=0; sy<options.patch_size; sy++, y+=options.patch_stride ) {
            if ( y < 0 || y >= rows ) continue;
            int x = j + options.patch_start_offset;
            for ( int sx=0; sx<options.patch_size; sx++, x+=options.patch_stride ) {
              if ( x < 0 || x >= cols ) continue;
              const dataType *data = (*this)(y,x);
              for ( int c=0; c<dimCell; c++ ) {
                sum += data[c] * data[c];
              }
            }
          }
          if ( sum > 1e-9 ) {
            inv_norm[k++] = 1.0 / sqrt(sum);
          } else {
            inv_norm[k++] = 0.0f;
          }
        }
      }
    }

    /* default constructor: zero sized image */
    FeatImage() : pyramid(1), inv_norm(1), scale_base(1.0), scales(0), scale_layer(1),
                  rows_layer(1), cols_layer(1),
                  rows(0), cols(0), id(-1), dimCell(0), options(0) 
    {
      scale_layer[0] = 1.0f;
      rows_layer[0] = rows;
      cols_layer[0] = cols;
      initPatchOptions();
    }

    /* empty image constructor */
    FeatImage( int h, int w, int dim ) : pyramid(1), scale_base(1.0), scales(0),
                                         scale_layer(1), rows_layer(1), cols_layer(1),
                                         rows(h), cols(w), id(-1), 
                                         dimCell(dim), options(dim)
    {
      initPatchOptions();
      pyramid.resize(1);
      pyramid[0].resize( h * w * dim, 0  );
      inv_norm.resize( h * w, 1.0 );
      memset( &pyramid[0][0], 0, sizeof(dataType) * h * w * dim );
      scale_layer[0] = 1.0f;
      rows_layer[0] = rows;
      cols_layer[0] = cols;
    }

    /* pyramid constructor */
    FeatImage( std::vector<FeatImage<dataType> > &&imgs, float base ) : pyramid(1), id(-1),
                                                                        dimCell(imgs[0].dimCell),
                                                                        options(dimCell)
                                                                        
    {
      initPatchOptions();
      if ( 0 == ( imgs.size() & 1 ) ) {
        Error( "FeatImage Constructor: expect odd number for pyramid levels." );
        exit( -1 );
      }

      scale_base = base;
      scales = static_cast<int>( imgs.size() >> 1 );
      rows = imgs[scales].rows;
      cols = imgs[scales].cols;
      dimCell = imgs[scales].dimCell;
      options = imgs[scales].options;
      
      scale_layer.resize( imgs.size() );
      scale_layer[scales] = 1.0;
      float scale = 1.0f;
      for ( int i=-1; i>=-scales; i-- ) {
        scale *= scale_base;
        scale_layer[scales+i] = scale;
      }

      scale = 1.0f;
      for ( int i=1; i<=scales;  i++ ) {
        scale /= scale_base;
        scale_layer[scales+i] = scale;
      }

      // fill rows_layer and cols_layer
      rows_layer.resize( imgs.size() );
      cols_layer.resize( imgs.size() );
      for ( int i=0, end=static_cast<int>( imgs.size() ); i<end; i++ ) {
        rows_layer[i] = imgs[i].rows;
        cols_layer[i] = imgs[i].cols;
      }

      
      pyramid.resize( imgs.size() );

      for ( int i=0, end=imgs.size(); i<end; i++ ) {
        pyramid[i].swap( imgs[i].pyramid[0] );
      }

      inv_norm.swap( imgs[scales].inv_norm );

    }
    
    /* move constructor */
    FeatImage( FeatImage<dataType> &&other ) : pyramid(std::move(other.pyramid)),
                                               inv_norm(std::move(other.inv_norm)),
                                               scale_base(other.scale_base),
                                               scales(other.scales),
                                               scale_layer(std::move(other.scale_layer)),
                                               rows_layer(std::move(other.rows_layer)),
                                               cols_layer(std::move(other.cols_layer)),
                                               rows(other.rows), cols(other.cols), id(other.id),
                                               dimCell(other.dimCell), options(other.options)
    {
    }


    /* move assignment */
    inline const FeatImage<dataType>& operator=( FeatImage<dataType> &&other )
    {
      rows = other.rows;
      cols = other.cols;
      id = other.id;
      dimCell = other.dimCell;
      pyramid.swap( other.pyramid );
      inv_norm.swap( other.inv_norm );
      scale_base = other.scale_base;
      scales = other.scales;
      scale_layer.swap( other.scale_layer );
      rows_layer.swap( other.rows_layer );
      cols_layer.swap( other.cols_layer );
      options = other.options;
      return (*this);
    }

    /* clone */
    inline FeatImage<dataType> clone() const
    {
      FeatImage<dataType> re;
      re.rows = rows;
      re.cols = cols;
      re.id = id;
      re.dimCell = dimCell;
      re.pyramid = pyramid;
      re.inv_norm = inv_norm;
      re.data = re.pyramid[0];
      re.options = options;
      return re;
    }

    
    
    /* ---------- Cell Accessors ---------- */

    inline const dataType* operator()( const int y, const int x ) const
    {
      return (&pyramid[scales][0]) + ( y * cols + x ) * dimCell;
    }


    inline const dataType* operator()( const int y, const int x, const int layer ) const
    {
      return (&pyramid[layer][0]) + ( y * cols_layer[layer] + x ) * dimCell;
    }

    inline const dataType* operator()( const int i ) const
    {
      return (&pyramid[scales][0]) + i * dimCell;
    }

    inline dataType* operator[]( const int i)
    {
      return (&pyramid[scales][0]) + i * dimCell;
    }
    
    inline dataType get( const int i ) const
    {
      return pyramid[scales][i];
    }

    inline dataType get( const int i, const int center ) const
    {
      return pyramid[scales][i] * inv_norm[center];
    }

    /* ---------- Patch Accessors ---------- */


    inline bool FetchPatch( int i, int j, dataType *feat ) const
    {

      int y = i + options.patch_start_offset;
      dataType *featp = feat;
      memset( feat, 0, sizeof(dataType) * options.patch_dim );

      bool re = true;
      for ( int l=0; l<options.patch_size; l++, y+=options.patch_stride ) {
        int x = j + options.patch_start_offset;
        for ( int k=0; k<options.patch_size; k++, x+=options.patch_stride ) {
          if ( 0 <= y && y < rows &&
               0 <= x && x < cols ) {
            memcpy( featp, (*this)(y,x), sizeof(dataType) * dimCell );
          } else {
            re = false;
          }
          featp += dimCell;
        }
      }

      
      if ( options.normalized ) {
        normalize_vec( feat, feat, options.patch_dim );
      }

      return re;

    }


    /* get patch from a particular layer of pyramid */
    inline bool FetchPatchLayer( int layer, float i, float j, float rotation, float scale, dataType *feat ) const
    {
      assert( layer >= 0 && layer < ( scales << 1 ) + 1 );
      
      float ratio = scale * scale_layer[layer];

      

      float cy = i * scale_layer[layer];
      float cx = j * scale_layer[layer];

      float cosa = cos( rotation );
      float sina = sin( rotation );
      
      float vert_y = cosa;
      float vert_x = -sina;
      float horz_y = -vert_x;
      float horz_x = vert_y;
      
      
      float y0 = cy + ( vert_y + horz_y ) * options.patch_start_offset;
      float x0 = cx + ( vert_x + horz_x ) * options.patch_start_offset;
      
      vert_y *= options.patch_stride;
      vert_x *= options.patch_stride;
      horz_y = -vert_x;
      horz_x = vert_y;

      dataType *vec0 = nullptr;
      dataType vec1[dimCell];

      dataType *featp = feat;
      bool re = true;

      for ( int l=0; l<options.patch_size; l++ ) {
        float y = y0;
        float x = x0;
        for ( int k=0; k<options.patch_size; k++ ) {

          int y1 = static_cast<int>( roundf( y ) );
          int x1 = static_cast<int>( roundf( x ) );

          bool b00 = 0 <= y1 && y1 < rows_layer[layer] && 0 <= x1 && x1 < cols_layer[layer];
          bool b10 = 0 <= y1 + 1 && y1 + 1 < rows_layer[layer] && 0 <= x1 && x1 < cols_layer[layer];
          bool b01 = 0 <= y1 && y1 < rows_layer[layer] && 0 <= x1 + 1 && x1 + 1 < cols_layer[layer];
          bool b11 = 0 <= y1 + 1 && y1 + 1 < rows_layer[layer] && 0 <= x1 + 1 && x1 + 1 < cols_layer[layer];

          float alpha = y - y1;
          float beta = x - x1;
          int indicator = 0;
          int miss = 0;

          // note from now on vec0 and featp are interchangebale
          vec0 = featp;
          
          if ( b00 && b10 ) {
            combine( (*this)( y1, x1, layer ), (*this)( y1 + 1, x1, layer ),
                     vec0, dimCell, 1.0f - alpha, alpha );
          } else if ( b00 ) {
            memcpy( vec0, (*this)( y1, x1, layer ), sizeof( dataType ) * dimCell );
          } else if ( b10 ) {
            memcpy( vec0, (*this)( y1 + 1, x1, layer ), sizeof( dataType ) * dimCell );
          } else {
            indicator = 1;
            miss++;
          }

          if ( b01 && b11 ) {
            combine( (*this)( y1, x1 + 1, layer ), (*this)( y1 + 1, x1 + 1, layer ),
                     vec1, dimCell, 1.0f - alpha, alpha );
          } else if ( b01 ) {
            memcpy( vec1, (*this)( y1, x1 + 1, layer ), sizeof( dataType ) * dimCell );
          } else if ( b11 ) {
            memcpy( vec1, (*this)( y1 + 1, x1 + 1, layer ), sizeof( dataType ) * dimCell );
          } else {
            indicator = -1;
            miss++;
          }

          if ( 2 == miss ) re = false;

          if ( 0 == indicator ) {
            combine( vec0, vec1, vec0, dimCell, 1.0f - beta, beta );
          } else if ( 1 == indicator ) {
            memcpy( vec0, vec1, sizeof( dataType ) * dimCell );
          } 

          y += horz_y;
          x += horz_x;
          featp += dimCell;
        }
        y0 += vert_y;
        x0 += vert_x;
      }

      if ( 0 != options.rotBins ) {
        float delta = rotation / options.shiftUnit;
        while ( delta < 0 ) delta += options.rotBins;
        dataType *featp = feat;
        for ( i=0; i<options.patch_dim; i+=options.rotBins ) {
          shift( featp, options.rotBins, delta );
          featp += options.rotBins;
        }
      }
      if ( options.normalized ) {
        normalize_vec( feat, feat, options.patch_dim );
      }

      return re;
    }

    
    /* get patch from a particular layer of pyramid */
    inline bool FetchPatch( int layer, float i, float j, float rotation, float scale, dataType *feat ) const
    {
      assert( layer >= 0 && layer < ( scales << 1 ) + 1 );

      float ratio = scale * scale_layer[layer];

      

      float cy = i * scale_layer[layer];
      float cx = j * scale_layer[layer];

      float cosa = cos( rotation );
      float sina = sin( rotation );

      float vert_y = cosa * ratio;
      float vert_x = -sina * ratio;
      float horz_y = -vert_x;
      float horz_x = vert_y;
      
      
      float y0 = cy + ( vert_y + horz_y ) * options.patch_start_offset;
      float x0 = cx + ( vert_x + horz_x ) * options.patch_start_offset;
      
      vert_y *= options.patch_stride;
      vert_x *= options.patch_stride;
      horz_y = -vert_x;
      horz_x = vert_y;

      dataType *vec0 = nullptr;
      dataType vec1[dimCell];

      dataType *featp = feat;
      bool re = true;

      for ( int l=0; l<options.patch_size; l++ ) {
        float y = y0;
        float x = x0;
        for ( int k=0; k<options.patch_size; k++ ) {

          int y1 = static_cast<int>( roundf( y ) );
          int x1 = static_cast<int>( roundf( x ) );


          bool b00 = 0 <= y1 && y1 < rows_layer[layer] && 0 <= x1 && x1 < cols_layer[layer];
          bool b10 = 0 <= y1 + 1 && y1 + 1 < rows_layer[layer] && 0 <= x1 && x1 < cols_layer[layer];
          bool b01 = 0 <= y1 && y1 < rows_layer[layer] && 0 <= x1 + 1 && x1 + 1 < cols_layer[layer];
          bool b11 = 0 <= y1 + 1 && y1 + 1 < rows_layer[layer] && 0 <= x1 + 1 && x1 + 1 < cols_layer[layer];


          
          float alpha = y - y1;
          float beta = x - x1;
          int indicator = 0;
          int miss = 0;

          // note from now on vec0 and featp are interchangebale
          vec0 = featp;
          
          if ( b00 && b10 ) {
            combine( (*this)( y1, x1, layer ), (*this)( y1 + 1, x1, layer ),
                     vec0, dimCell, 1.0f - alpha, alpha );
          } else if ( b00 ) {
            memcpy( vec0, (*this)( y1, x1, layer ), sizeof( dataType ) * dimCell );
          } else if ( b10 ) {
            memcpy( vec0, (*this)( y1 + 1, x1, layer ), sizeof( dataType ) * dimCell );
          } else {
            indicator = 1;
            miss++;
          }

          if ( b01 && b11 ) {
            combine( (*this)( y1, x1 + 1, layer ), (*this)( y1 + 1, x1 + 1, layer ),
                     vec1, dimCell, 1.0f - alpha, alpha );
          } else if ( b01 ) {
            memcpy( vec1, (*this)( y1, x1 + 1, layer ), sizeof( dataType ) * dimCell );
          } else if ( b11 ) {
            memcpy( vec1, (*this)( y1 + 1, x1 + 1, layer ), sizeof( dataType ) * dimCell );
          } else {
            indicator = -1;
            miss++;
          }

          if ( 2 == miss ) re = false;

          if ( 0 == indicator ) {
            combine( vec0, vec1, vec0, dimCell, 1.0f - beta, beta );
          } else if ( 1 == indicator ) {
            memcpy( vec0, vec1, sizeof( dataType ) * dimCell );
          } 

          y += horz_y;
          x += horz_x;
          featp += dimCell;
        }
        y0 += vert_y;
        x0 += vert_x;
      }

      if ( 0 != options.rotBins ) {
        float delta = rotation / options.shiftUnit;
        while ( delta < 0 ) delta += options.rotBins;
        dataType *featp = feat;
        for ( i=0; i<options.patch_dim; i+=options.rotBins ) {
          shift( featp, options.rotBins, delta );
          featp += options.rotBins;
        }
      }
      if ( options.normalized ) {
        normalize_vec( feat, feat, options.patch_dim );
      }

      return re;
    }
    

    /* rotation is clockwize and in rad
     * scale > 1 = upsampled and < 1 = downsampled
     */
    inline bool FetchPatch( float i, float j, float rotation, float scale, dataType *feat ) const
    {

      float log_scale = log(scale) / log(scale_base);
      if ( 0 == scales ) {
        return FetchPatch( 0, i, j, rotation, scale, feat );
      } else if ( log_scale < -scales ) {
        return FetchPatch( -scales, i, j, rotation, scale, feat );
      } else if ( log_scale > scales ) {
        return FetchPatch( scales, i, j, rotation, scale, feat );
      } else {
        int l = static_cast<int>( log_scale );
        float alpha = log_scale - l;
        dataType tmp[options.patch_dim];
        bool a = FetchPatch( l + scales, i, j, rotation, scale, feat );
        bool b = FetchPatch( l + 1 + scales, i, j, rotation, scale,tmp );
        combine( feat, tmp, feat, options.patch_dim, 1 - alpha, alpha );
        return a && b;
      }
    }

    
    inline PatchProxy Spawn( int i, int j ) const 
    {
      return PatchProxy( this, i, j );
    }
    
    

    /* ---------- Operations ---------- */
    inline void MeanFilter( int wndRadius = 1 )
    {
      // Row Integaral
      dataType *tmp = new dataType[rows * cols * dimCell];
      for ( int i=0; i<rows; i++ ) {
        dataType *fastp = tmp + i * cols * dimCell;
        dataType *slowp = fastp;
        dataType *dp = &pyramid[scales][0] + i * cols * dimCell;
        for ( int c=0; c<dimCell; c++ ) {
          *(fastp++) = *(dp++);
        }
        for ( int j=1; j<cols; j++ ) {
          for ( int c=0; c<dimCell; c++ ) {
            *(fastp++) = *(dp++) + *(slowp++);
          }
        }
      }

      // Column Integral
      for ( int j=0; j<cols; j++ ) {
        dataType *slowp = tmp + j * dimCell;
        dataType *fastp = tmp + ( j + cols ) * dimCell;

        for ( int i=1; i<rows; i++ ) {
          for ( int c=0; c<dimCell; c++ ) {
            *(fastp++) += *(slowp++);
          }
          fastp += (cols-1) * dimCell;
          slowp += (cols-1) * dimCell;
        }
      }

      // mean calculation
      for ( int i=0; i<rows; i++ ) {
        for ( int j=0; j<cols; j++ ) {
          int top = i - wndRadius - 1;
          if ( top < 0 ) top = -1;
          int left = j - wndRadius - 1;
          if ( left < 0 ) left = -1;
          int bottom = i + wndRadius;
          if ( bottom >= rows ) bottom = rows - 1;
          int right = j + wndRadius;
          if ( right >= cols ) right = cols - 1;

          for ( int c=0; c<dimCell; c++ ) {
            dataType sum = tmp[ ( bottom * cols + right ) * dimCell + c ];
            if ( left >  -1 ) sum -= tmp[ ( bottom * cols + left ) * dimCell + c ];
            if ( top > -1 ) sum -= tmp[ ( top * cols + right ) * dimCell + c ];
            if ( left > -1 && top > -1 ) sum += tmp[ ( top * cols + left ) * dimCell + c ];
            int area = ( bottom - top ) * ( right - left );

            pyramid[scales][ ( i * cols + j ) * dimCell + c ] = sum / area;
          }
        }
      }

      DeleteToNullWithTestArray( tmp );
      
    }

    inline FeatImage<dataType> Group( int wndRadius ) const
    {
      int side = ((wndRadius << 1) + 1);
      int area = side * side;
      
      FeatImage<dataType> img( rows, cols, area * dimCell );
      for ( int i=0; i<rows; i++ ) {
        for ( int j=0; j<cols; j++ ) {
          
          dataType *ptr = img[ i * cols + j ];
          memset( ptr, 0, sizeof(dataType) * area * dimCell );
          for ( int dy=-wndRadius; dy<=wndRadius; dy++ ) {
            int y = i + dy;
            for ( int dx=-wndRadius; dx<=wndRadius; dx++ ) {
              int x = j + dx;
              if ( 0 <= y && y < rows && 0 <= x && x < cols ) {
                memcpy( ptr, (*this)(y,x), sizeof(dataType) * dimCell );
              }
              ptr += dimCell;
            }
          }

        }
      }

      return img;
    }

    
    inline void NormalizeCell()
    {
      for ( int i=0, end=rows*cols; i<end; i++ ) {
        normalize_vec<dataType>( (*this)[i], (*this)[i], dimCell );
      }
    }
    
    
    /* ---------- Operators ---------- */
    // inline bool operator==( const FeatImage<dataType> &other )
    // {
    //   if ( rows != other.rows || cols != other.cols || dimCell != other.dimCell ) {
    //     return false;
    //   }

    //   for ( int i=0; i<rows * cols * dimCell; i++ ) {
    //     if ( abs( data[i] - other.data[i] ) > 1e-5 ) {
    //       return false;
    //     }
    //   }

    //   return true;
    // }

    /* ---------- Tracer ---------- */
    template <typename T = dataType>
    inline void showCell( const int i, ENABLE_IF( (!std::is_same<T,unsigned char>::value) )  ) const
    {
      const dataType* f = (*this)(i);
      std::cout << "( ";
      for ( int c=0; c<dimCell; c++ ) {
        std::cout << f[c] << " ";
      }
      std::cout << ")\n";
    }

    template <typename T = dataType>
    inline void showCell( const int i, ENABLE_IF((std::is_same<T,unsigned char>::value))  ) const
    {
      const dataType* f = (*this)(i);
      printf( "( " );
      for ( int c=0; c<dimCell; c++ ) {
        printf( "%hhu ", f[c] );
      }
      printf( ")\n" );
    }

    template <typename T = dataType>
    inline void showCell( const int i, const int j, ENABLE_IF((!std::is_same<T,unsigned char>::value)) ) const
    {
      const dataType* f = (*this)(i,j);
      std::cout.precision(5);
      std::cout << "( ";
      for ( int c=0; c<dimCell; c++ ) {
        std::cout << f[c] << " ";
      }
      std::cout << ")\n";
    }

    template <typename T = dataType>
    inline void showCell( const int i, const int j, ENABLE_IF((std::is_same<T,unsigned char>::value))  ) const
    {
      const dataType* f = (*this)(i,j);
      printf( "( " );
      for ( int c=0; c<dimCell; c++ ) {
        printf( "%hhu ", f[c] );
      }
      printf( ")\n" );
    }


  };


  // +-------------------------------------------------------------------------------
  // | Album, collection of imges.
  // | Serve as owner of images, as well as owner of patches.
  // +-------------------------------------------------------------------------------
  template <typename dataType>
  class Album
  {
  private:
    std::vector< FeatImage<dataType> > pages;
    Album( const Album<dataType>& other );
  public:

    /* ---------- constructors ---------- */
    
    Album()
    {
      pages.clear();
    }

    /* move constructor */
    Album( Album<dataType>&& other )
    {
      pages.swap( other.pages );
    }

    /* push() has side effect. It is destructive as it steals the
     * FeatImage that img refer to. */
    inline void push( FeatImage<dataType>&& img )
    {
      img.id = static_cast<int>( pages.size() );
      pages.push_back( std::move(img) );
    }

    // create a vector of patches
    std::vector<typename FeatImage<dataType>::PatchProxy> list( int margin = 7, int stride = 1 )
    {
      std::vector<typename FeatImage<dataType>::PatchProxy> res;
      for ( auto& img : pages ) {
        for ( int i=margin; i<img.rows-margin; i+=stride ) {
          for ( int j=margin; j<img.cols-margin; j+=stride ) {
            res.push_back( img.Spawn( i, j ) );
          }
        }
      }
      return res;
    }


    inline void SetPatchSize( int s )
    {
      for ( auto& img : pages ) {
        img.SetPatchSize( s );
      }
    }
      
    inline void SetPatchStride( int s )
    {
      for ( auto& img : pages ) {
        img.SetPatchStride( s );
      }
    }

    /* Only provide read-only access to member images */
    const FeatImage<dataType>& operator()( const int index ) const
    {
      return pages[index];
    }

    int size() const
    {
      return static_cast<int>( pages.size() );
    }


    inline typename std::vector<FeatImage<dataType> >::const_iterator begin()
    {
      return pages.begin();
    }

    inline typename std::vector<FeatImage<dataType> >::const_iterator end()
    {
      return pages.end();
    }

  };
  
}
