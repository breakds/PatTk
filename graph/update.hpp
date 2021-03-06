/*********************************************************************************
 * File: update.hpp
 * Description: update the candiadte set of a certain image
 * by BreakDS, University of Wisconsin Madison, Sat Aug 18 16:24:17 CDT 2012
 *********************************************************************************/

#pragma once
#include <cassert>
#include <string>
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/SafeOP.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "Graph.hpp"
#include "../data/2d.hpp"
#include "../data/features.hpp"
#include "../opt/BP.hpp"
#include "../opt/BP_CUDA.h"



// Temporary Constants
#define PATCH_SIDE 17

#define ENABLE_CUDA 1

#define OPTIMIZE 1

using namespace EnvironmentVariable;

namespace PatTk
{
  namespace
  {
    void GenConfDefault( const std::string &directory, const std::string &target,
                         const std::string &reference )
    {
      WITH_OPEN( out, "PatchMatch.conf", "w" );
      fprintf( out, "(include pm_common.conf)\n" );
      fprintf( out, "(directory %s)\n", directory.c_str() );
      fprintf( out, "(target %s)\n", target.c_str() );
      fprintf( out, "(files %s)\n", reference.c_str() );
      fprintf( out, "(algorithm rotscale)\n" );
      fprintf( out, "(patch-w 9)\n" );
      fprintf( out, "(nn-iters 5)\n" );
      fprintf( out, "(rs-max -1)\n" );
      fprintf( out, "(rs-min -1)\n" );
      fprintf( out, "(rs-ratio -1)\n" );
      fprintf( out, "(rs-iters 5)\n" );
      fprintf( out, "(cores 6)\n" );
      fprintf( out, "(bmask null)\n" );
      fprintf( out, "(window-h -1)\n" );
      fprintf( out, "(window-w -1)\n" );
      fprintf( out, "(ann-prev null)\n" );
      fprintf( out, "(ann-window null)\n" );
      fprintf( out, "(ann-winsize null)\n" );
      fprintf( out, "(scalerange 2)\n" );
      END_WITH( out );
    }

    void GenConfTemp( int targetID, int refID )
    {
      system( "cp -f PatchMatch.conf.template PatchMatch.conf" );
      WITH_OPEN( out, "PatchMatch.conf", "a" );
      fprintf( out, "(target %d)\n", targetID );
      fprintf( out, "(source %d)\n", refID );
      END_WITH( out );
    }

    PatGraph GetMapping( const std::string &filename, int h, int w, int index )
    {
      static const int K = env["graph-degree"];
      PatGraph graph( h, w );
      WITH_OPEN( in, filename.c_str(), "r" );
      // Assume all the pixels has candiadtes (stride 1 for patchmatch)
      // Note Zhouyuan use column-major in his output file
      int tmp = 0;
      for ( int j=0; j<w; j++ ) {
        for ( int i=0; i<h; i++ ) {
          int x, y;
          FSCANF_CHECK( in, "%d", &x ); // Skip x for target patch
          FSCANF_CHECK( in, "%d", &y ); // Skip y for target patch
          for ( int k=0; k<K; k++ ) {
            PatLoc loc;
            loc.index = index;
            FSCANF_CHECK( in, "%d", &tmp );
            
            float fTmp = 0;
            FSCANF_CHECK( in, "%f", &fTmp );
            loc.x = static_cast<int>( fTmp );
            FSCANF_CHECK( in, "%f", &fTmp );
            loc.y = static_cast<int>( fTmp );

            FSCANF_CHECK( in, "%f", &loc.dist );
            if ( 480000.0 <= loc.dist ) loc.dist = 480000.0f; // set an upperbound for dist
            FSCANF_CHECK( in, "%f", &loc.rotation );
            FSCANF_CHECK( in, "%f", &loc.scale );
            graph[i*w+j].push_back( loc );
          }
        }
      }
      END_WITH( in );
      return graph;
    }

    template <typename floating=float>
    class RandProj : public optimize::AbstractRandHash<floating>
    {
    public:
      vector<floating> coeff;
      void shuffle( int dim ) 
      {
        static const float ub = static_cast<double>( RAND_MAX );
        coeff.resize(dim);
        for ( int i=0; i<dim; i++ ) {
          coeff[i] = rand() / ub;
        }
      }
  
      floating operator()( const floating* a, int dim )
      {
        floating sum = 0.0;
        for ( int i=0; i<dim; i++ ) {
          sum += a[i] * coeff[i];
        }
        return sum;
      }
    };

  }

  template <typename valueType=float>
  class FakeLabelDist : public optimize::AbstractDistMetric<valueType>
  {
  private:
    valueType coeff[6];
  public:
    FakeLabelDist() 
    {
      coeff[0] = 0.0;
      coeff[1] = 30.0;
      coeff[2] = 30.0;
      coeff[3] = 10.0;
      coeff[4] = 1.0;
      coeff[5] = 1.0;
    }
    

    // inline valueType operator()( const valueType *a, const valueType *b,
    //                              int __attribute__((__unused__)) dim,
    //                              int direction=-1 )
    // {
    //   valueType tmp;
      
    //   // [0] = image index
    //   tmp = ( a[0] > b[0] ) ? ( a[0] - b[0] ) : ( b[0] - a[0] );
    //   if ( tmp > 1.0 ) return static_cast<valueType>( 150000.0 );

    //   // [1], [2] = dy, dx (rotation representation)
    //   tmp = ( a[1] > b[1] ) ? ( a[1] - b[1] ) : ( b[1] - a[1] );
    //   tmp += ( a[2] > b[2] ) ? ( a[2] - b[2] ) : ( b[2] - a[2] );
    //   if ( tmp > 1.0 ) {
    //     return static_cast<valueType>( 150000.0 );
    //   }

      
    //   // [4],[5] = spatial distance
    //   // Should be compensated by the rotation (dy,dx)
    //   // Not that dy = b[1] dx = b[2]
    //   valueType ay(a[4]), ax(a[5]);
    //   if ( 0 == direction ) {
    //     ay -= b[2];
    //     ax -= b[1];
    //   } else if ( 1 == direction ) {
    //     ay += b[1];
    //     ax -= b[2];
    //   } else if ( 2 == direction ) {
    //     ay += b[2];
    //     ax += b[1];
    //   } else if ( 3 == direction ) {
    //     ay -= b[1];
    //     ax += b[2];
    //   }
      
    //   tmp = ( ay > b[4] ) ? ( ay - b[4] ) : ( b[4] - ay );
    //   tmp += ( ax > b[5] ) ? ( ax - b[5] ) : ( b[5] - ax );
    //   if ( tmp > PATCH_SIDE ) {
    //     return static_cast<valueType>( 150000.0 );
    //   }
      
    //   // [1],[2] = spatial distance

    //   valueType sum = tmp * coeff[4];
    //   for ( int i=1; i<4; i++ ) {
    //     sum += ( ( a[i] > b[i] ) ? ( a[i] - b[i] ) : ( b[i] - a[i] ) ) * coeff[i];
    //   }
    //   return sum;
    // }

    inline valueType operator()( const valueType *a, const valueType *b,
                                 int __attribute__((__unused__)) dim,
                                 int __attribute__((__unused__)) direction=-1 )
    {
      valueType tmp;
      
      // [0] = image index
      tmp = ( a[0] > b[0] ) ? ( a[0] - b[0] ) : ( b[0] - a[0] );
      if ( tmp > 1.0 ) return static_cast<valueType>( 150000.0 );

      // [1], [2] = dy, dx (rotation representation)
      tmp = ( a[1] > b[1] ) ? ( a[1] - b[1] ) : ( b[1] - a[1] );
      tmp += ( a[2] > b[2] ) ? ( a[2] - b[2] ) : ( b[2] - a[2] );
      if ( tmp > 1.5 ) {
        return static_cast<valueType>( 150000.0 );
      }

      tmp = 0.0f;
          
      tmp += fabsf( b[1] - a[1] ) * coeff[1];
      tmp += fabsf( b[2] - a[2] ) * coeff[2];
      tmp += fabsf( b[4] - a[4] ) * coeff[4];
      tmp += fabsf( b[5] - a[5] ) * coeff[5];

      return tmp;
    }
    
  };


  void BuildArrays( float* &D, float* &label, const PatGraph &graph, int candNum,
                    const Album<BGRCell,int,false> &album, const int targetID ) 
  {

    const Image<BGRCell,int,false> &img = album(targetID);

    int tarH = graph.rows;
    int tarW = graph.cols;

    DeleteToNullWithTestArray( D );
    D = new float[ tarH * tarW * candNum ];
    float *Dp = D;
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        Image<BGRCell,int,false>::Patch p0 = img.Spawn( i, j, 1.0, 0.0 );
        // assume all candidate set is of size K * 2
        for ( int k=0; k<candNum; k++ ) {
          *(Dp++) = sqrt( graph(i,j)[k].dist );
        }      
      }
    }

    DeleteToNullWithTestArray( label );
    label = new float[tarH * tarW * candNum * 6];
    float *labelp = label;
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        for ( int k=0; k<candNum; k++ ) {
          graph(i,j)[k].GetTransform( labelp, i, j );
          labelp += 6;
          // *(labelp++) = graph(i,j)[k].index;
          // *(labelp++) = sin(graph(i,j)[k].rotation);
          // *(labelp++) = cos(graph(i,j)[k].rotation);
          // *(labelp++) = graph(i,j)[k].scale;
          // *(labelp++) = graph(i,j)[k].y;
          // *(labelp++) = graph(i,j)[k].x;
        }
      }
    }
  }

  
  int Enrichment( PatGraph &graph, int* result, int origCandNum,
                  const Album<BGRCell,int,false> &album, const int targetID ) 
  {
    static const int num = 5;
    static const int di[5] = { -1, 0, 1, 0, 0 };
    static const int dj[5] = { 0, -1, 0, 1, 0 };
    const Image<BGRCell,int,false> &img = album(targetID);


    Info( "Enrichment started ..." );
    // Construct Enrichment graph
    PatGraph enrichment( graph.rows, graph.cols );
    int k = -1;
    for ( int i=0; i<graph.rows; i++ ) {
      for ( int j=0; j<graph.cols; j++ ) {
        k++;
        for ( int dir=0; dir<5; dir++ ) {
          int i1 = i + di[dir];
          int j1 = j + dj[dir];
          if ( i1 < 0 || i1 >= graph.rows ||
               j1 < 0 || j1 >= graph.cols ) {
            continue;
          }
          int d = i1 * graph.cols + j1;
          enrichment[d].push_back( PatLoc( graph(k)[result[k]], di[dir], dj[dir] ) );
          // calculate distance
          int sum = 0;
          int tmp = 0;
          auto p0 = img.Spawn( i1, j1, 1.0, 0.0 );
          auto p1 = enrichment(d).back().toPatch( album );
          for ( int c=0, end=p0.dim(); c<end; c++ ) {
            tmp = p1[c] - p0[c];
            sum += tmp * tmp;
          }
          enrichment[d].back().dist = static_cast<float>( sum );
        }
      }
    }
    

    for ( int i=0; i<graph.rows*graph.cols; i++ ) {
      // fill and compensation
      graph[i].resize( num + origCandNum );
      for ( int k=0; k<num; k++ ) {
        if ( k < static_cast<int>( enrichment[i].size() ) ) {
          graph[i][k+origCandNum] = enrichment[i][k];
        } else {
          graph[i][k+origCandNum] = enrichment[i].back();
        }
      }
    }

    Done( "Enrichment" );

    return (num+origCandNum);
  }

  double UpdateGraph( const std::vector<std::string> &imgList, // filenames for all the images
                      const Album<BGRCell,int,false> &album,
                      const int tarH, // width of the target
                      const int tarW, // height of the target
                      const int targetID, // image id of the target
                      const int referenceID ) // image id of the reference
  {

    // Constants
    static const float lambda = 5.00;
    static const int K = env["gen"];
    static const int num = 5;
    int area = tarH * tarW;


    
    // Generate the configuration file
    // GenConfDefault( env["directory"], imgList[targetID], imgList[referenceID] );
    GenConfTemp( targetID, referenceID );
    
    // Call nnmex externally
    // system( "./nnmex PatchMatch.conf" );
    
    
    string mappingPath = "./mapping.dat";
    PatGraph graph( mappingPath );


    // Old Graph:
    PatGraph graphOld( tarH, tarW );
    string oldpath = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[targetID].c_str() );
    int candNum = K;
    if ( probeFile(oldpath) ) {
      graphOld = std::move( PatGraph(oldpath) );
      candNum = K + env["keep"];
    };

    // Merge Graphs
    graph += graphOld;

    // Prepare data term ( height x width x K );
    float *D = nullptr;
    
    // Prepare labels term ( height x width x K x dim ) (dim=6)
    float *label = nullptr;

    // Prepare the result array
    int result[area];

    // Prepare the message array
    float *msg = new float[tarH*tarW*(candNum+num)*4];
    
    int enrichIter = env["enrich-iter"];

    memset( result, 0, sizeof(int) * area );
    int labelNum = candNum;

    double energy = 0.0;


    for ( int iter=0; iter<enrichIter; iter++ ) {

      BuildArrays( D, label, graph, labelNum, album, targetID );

      printf( "candNum = %d\n", labelNum );

      printf( "Shell Iteration %d:\n", iter );
#ifndef ENABLE_CUDA
      // Loopy BP
      optimize::Options options;
      options.maxIter = 10;
      options.numHypo = 3;
      options.verbose = 0;

      timer::tic();


      energy = optimize::LoopyBP<FakeLabelDist<float>, float>( D, label, lambda, 
                                                                      tarH, tarW, labelNum, 6,
                                                                      result, options, msg );
      printf( "BP is done. time elapsed: %.2lf sec\n", timer::utoc() );
#else
      // Loopy BP Cuda Version
      optimize_cuda::Options options;
      options.maxIter = 10;
      options.verbose = 0;
      options.lambda = lambda;

      timer::tic();

      energy = optimize_cuda::LoopyBP( D, label, tarH, tarW, labelNum, 6, result, options, msg );
      
      printf( "BP is done. time elapsed: %.2lf sec\n", timer::utoc() );
#endif
      if ( iter < enrichIter - 1 ) { 
        labelNum = Enrichment( graph, result, candNum, album, targetID );
      }
    }
    
    
    // Save Optimization Result
    PatGraph resGraph( graph.rows, graph.cols );
    for ( int i=0; i<area; i++ ) {
      resGraph[i].push_back( graph(i)[result[i]] );
    }
    string savepath = strf( "%s/%s.res", env["graph-dir"].c_str(), imgList[referenceID].c_str() );
    resGraph.write( savepath );



    
    // eliminate the bottom candidates
    int keep = env["keep"];
    for ( int i=0; i<area; i++ ) {
      heap<float,int> ranker( keep );
      for ( int k=0; k<candNum; k++ ) {
        float key = D[i*candNum+k];
        for ( int d=0; d<4; d++ ) {
          key += msg[(area*d+i)*candNum+k];
        }
        ranker.add( key, k );
      }
      vector<PatLoc> tmp;
      tmp.reserve( K );
      for ( int j=0; j<keep; j++ ) {
        tmp.push_back( graph(i)[ranker[j]] );
      }
      graph[i].swap( tmp );
    }

    // Save temporary candidates mapping
    savepath = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[targetID].c_str() );
    graph.write( savepath );
    
    
    // write result out
    WITH_OPEN( out, "result.dat", "w" );
    fwrite( &graph.rows, sizeof(int), 1, out );
    fwrite( &graph.cols, sizeof(int), 1, out );
    fwrite( &result, sizeof(int), graph.rows * graph.cols, out );
    END_WITH( out );

    
    DeleteToNullWithTestArray( label );
    DeleteToNullWithTestArray( msg );

    return energy;
  }
};
