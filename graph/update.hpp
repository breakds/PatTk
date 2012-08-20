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
          FSCANF_CHECK( in, "%d", &tmp ); // Skip x for target patch
          FSCANF_CHECK( in, "%d", &tmp ); // Skip y for target patch
          for ( int k=0; k<K; k++ ) {
            PatLoc loc;
            loc.index = index;
            FSCANF_CHECK( in, "%d", &tmp );
            FSCANF_CHECK( in, "%f", &loc.x );
            FSCANF_CHECK( in, "%f", &loc.y );
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


  
  void UpdateGraph( const std::vector<std::string> &imgList, // filenames for all the images
                    Album<HoGCell,int,false> &album,
                    int targetID, // image id of the target
                    int referenceID ) // image id of the reference
  {
    assert( album.size() == static_cast<int>( imgList.size() ) );
    
    // Constants
    static const float lambda = 1.0;
    static const int K = env["graph-degree"];
    int tarH = album(targetID).rows;
    int tarW = album(targetID).cols;
    int area = tarH * tarW;
    
    // Generate the configuration file
    GenConfDefault( env["directory"], imgList[targetID], imgList[referenceID] );
    
    // Call nnmex externally
    // system( "./nnmex PatchMatch.conf" );
    
    
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
    


    for ( auto& ele : graph(0,89) ) {
      ele.show();
    }
    
    
    // Prepare data term ( height x width x K );
    float D[tarH * tarW * candNum ];
    float *Dp = D;
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        // assume all candidate set is of size K * 2
        for ( int k=0; k<candNum; k++ ) {
          *(Dp++) = graph(i,j)[k].dist;
        }      
      }
    }
    
    // Prepare labels term ( height x width x K x dim ) (dim=5)
    float *label = new float[tarH * tarW * candNum * 6];
    float *labelp = label;
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        for ( int k=0; k<candNum; k++ ) {
          *(labelp++) = graph(i,j)[k].index * 15000.0;
          *(labelp++) = sin(graph(i,j)[k].rotation) * 300.0;
          *(labelp++) = cos(graph(i,j)[k].rotation) * 300.0;
          *(labelp++) = graph(i,j)[k].scale * 100.0;
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
    
    optimize::LoopyBP<RandProj<float>, optimize::FDT<float>, float>( D, label, lambda, 
                                                                     tarH, tarW, candNum, 6,
                                                                     result, options );
    
    for ( int i=0; i<tarH; i++ ) {
      for ( int j=0; j<tarW; j++ ) {
        printf( "(%d,%d)->(%d | %.2f,%.2f) with scale %.2f, rotation %.2f, dist=%.4f -> picked = %d.\n", i, j,
                graph(i,j)[result[i*tarW+j]].index,
                graph(i,j)[result[i*tarW+j]].y,
                graph(i,j)[result[i*tarW+j]].x,
                graph(i,j)[result[i*tarW+j]].scale,
                graph(i,j)[result[i*tarW+j]].rotation,
                graph(i,j)[result[i*tarW+j]].dist,
                result[i*tarW+j] );
        char ch;
        scanf( "%c", &ch );
      }
    }
    

    // Save candidates
    // string savepath = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[targetID].c_str() );
    // graph.write( savepath );
    
    DeleteToNullWithTestArray( label );
  }
};
