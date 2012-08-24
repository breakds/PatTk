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
      fprintf( out, "(patch-w 17)\n" );
      fprintf( out, "(nn-iters -1)\n" );
      fprintf( out, "(rs-max -1)\n" );
      fprintf( out, "(rs-min -1)\n" );
      fprintf( out, "(rs-ratio -1)\n" );
      fprintf( out, "(rs-iters -1)\n" );
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
  public:
    inline valueType operator()( const valueType *a, const valueType *b, int dim )
    {
      valueType tmp;
      tmp = ( a[0] > b[0] ) ? ( a[0] - b[0] ) : ( b[0] - a[0] );
      if ( tmp > 1.0 ) return static_cast<valueType>( 2000.0 );

      valueType sum = 0;
      for ( int i=1; i<dim; i++ ) {
        sum += ( a[i] > b[i] ) ? ( a[i] - b[i] ) : ( b[i] - a[i] );
      }
      return sum;
    }
  };


  
  void UpdateGraph( const std::vector<std::string> &imgList, // filenames for all the images
                    const int tarH, // width of the target
                    const int tarW, // height of the target
                    const int targetID, // image id of the target
                    const int referenceID ) // image id of the reference
  {
    // Constants
    static const float lambda = 2.0;
    static const int K = env["graph-degree"];
    int area = tarH * tarW;
    
    // Generate the configuration file
    GenConfDefault( env["directory"], imgList[targetID], imgList[referenceID] );
    
    // Call nnmex externally
    system( "./nnmex PatchMatch.conf" );
    
    
    // New Graph:
    PatGraph graphNew = std::move( GetMapping( strf( "%s/mapping.txt", env["directory"].c_str() ), 
                                               tarH, tarW, referenceID ) );
    
    // Old Graph:
    PatGraph graph( tarH, tarW );
    string oldpath = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[targetID].c_str() );
    int candNum = K;
    if ( probeFile(oldpath) ) {
      graph = std::move( PatGraph(oldpath) );
      candNum = K << 1;
    };


    
    
    
    // Merge Graphs
    graph += graphNew;




    
    
    // Prepare data term ( height x width x K );
    float D[tarH * tarW * candNum ];
    float *Dp = D;
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        // assume all candidate set is of size K * 2
        for ( int k=0; k<candNum; k++ ) {
          *(Dp++) = sqrt( graph(i,j)[k].dist );
        }      
      }
    }

    // Prepare labels term ( height x width x K x dim ) (dim=6)
    float *label = new float[tarH * tarW * candNum * 6];
    float *labelp = label;
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        for ( int k=0; k<candNum; k++ ) {
          *(labelp++) = graph(i,j)[k].index;
          *(labelp++) = sin(graph(i,j)[k].rotation) * 30.0;
          *(labelp++) = cos(graph(i,j)[k].rotation) * 30.0;
          *(labelp++) = graph(i,j)[k].scale * 10.0;
          *(labelp++) = graph(i,j)[k].y;
          *(labelp++) = graph(i,j)[k].x;
        }
      }
    }

    // Prepare the result array
    int result[area];

    // Loopy BP
    optimize::Options options;
    options.maxIter = 10;
    options.numHypo = 3;
    options.verbose = 1;

    printf( "candNum = %d\n", candNum );

    timer::tic();
    float *msg = new float[tarH*tarW*candNum*4];
    optimize::LoopyBP<FakeLabelDist<float>, float>( D, label, lambda, 
                                                    tarH, tarW, candNum, 6,
                                                    result, options, msg );
    printf( "BP is done. time elapsed: %.2lf sec\n", timer::utoc() );
    

    // for ( int i=0; i<tarH; i++ ) {
    //   for ( int j=0; j<tarW; j++ ) {
    //     printf( "(%d,%d)->(%d | %d,%d) with scale %.2f, rotation %.2f, dist=%.4f -> picked = %d.\n", i, j,
    //             graph(i,j)[result[i*tarW+j]].index,
    //             graph(i,j)[result[i*tarW+j]].y,
    //             graph(i,j)[result[i*tarW+j]].x,
    //             graph(i,j)[result[i*tarW+j]].scale,
    //             graph(i,j)[result[i*tarW+j]].rotation,
    //             graph(i,j)[result[i*tarW+j]].dist,
    //             result[i*tarW+j] );
    //     char ch;
    //     scanf( "%c", &ch );
    //   }
    // }
    
    

    // eliminate the bottom candidates
    for ( int i=0; i<area; i++ ) {
      heap<float,int> ranker( K );
      for ( int k=0; k<candNum; k++ ) {
        float key = 0.0;
        for ( int d=0; d<4; d++ ) {
          key += msg[(area*d+i)*K+k];
        }
        ranker.add( key, k );
      }
      vector<PatLoc> tmp;
      tmp.reserve( K );
      for ( int j=0; j<K; j++ ) {
        tmp.push_back( graph(i)[ranker[j]] );
      }
      graph[i].swap( tmp );
    }
 

    // Save candidates
    string savepath = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[targetID].c_str() );
    graph.write( savepath );
    
    
    
    
    DeleteToNullWithTestArray( label );
    DeleteToNullWithTestArray( msg );
  }
};
