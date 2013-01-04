#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/SafeOP.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/forest.hpp"
#include "../graph/Trans.hpp"
#include "../graph/Bipartite.hpp"



using namespace PatTk;
using namespace EnvironmentVariable;





class Solver {
  /* Solver for objective function
   *           \sum_m      ( \sum_l alpha(m,l) * q(l) - P(m) )^2
   *  + \beta  \sum_{i,j}  w(i,j) [ \sum_l (alpha(i,l)-alpha(j,l)) q(l) ]^2
   *
   *  where
   *  1. P(m)           -    the groud truth label distribution for patch m
   *  2. alpha(m,l)     -    coefficients for voter l against patch m
   *  3. q(l)           -    the self-voting label distribution of voter l
   *  4. beta           -    coefficient for the regularization term
   *  5. w(i,j)         -    feature similarity between patch i and patch j
   *
   *  and
   *  1. D(m)           =    \sum_l alpha(m,l) * q(l) - P(m)
   *  2. d(n)           =    \sum_l (alpha(i_n,l) - alpha(j_n,l)) q(l)
   *  2. numL           -    number of labeled patches
   *  3. numU           -    number of unlabeled patches
   *  4. K              -    number of classes
   *  5. L              -    number of voters
   *  6. N              -    number of patch neighbor pairs
   */
public:
  struct Options
  {
    double beta; // coefficient for the pairwise regularization term
    int maxIter; // iteration number
    double shrinkRatio; // Shrinking Ratio for line search
    double wolf; // Sufficent descreasing condition coefficient

    Options()
    {
      beta = 1.0;
      maxIter = 10;
      shrinkRatio = 0.8;
      wolf = 0.999;
    }
  } options;

private:

  const double *P;
  const double *w;

  
  double *q;

  double *D;
  double *d;
  
  int K, L, N, numL, numU;
  
  const Bipartite *m_to_l;
  const std::vector<std::pair<int,int> > *patchPairs;


public:
  
  // constructor
  Solver()
  {
    D = nullptr;
    P = nullptr;
    q = nullptr;
    d = nullptr;
    w = nullptr;
    
    K = 0;
    L = 0;
    N = 0;
    numL = 0;
    numU = 0;
    
    m_to_l = nullptr;
    patchPairs = nullptr;
  }
  
private:
  
  

  /* ---------- D(m) ---------- */

  // D(m) = sum alpha(l,m)*q(l,m) - P(m)
  inline void update_D( int m )
  {
    auto& _to_l = m_to_l->getToSet( m );

    // Set D(m) to 0
    double *t = D + m * K;
    zero( t, K );

    // Note t is an alias of D(m) now
    
    // add q(l) * alpha(m,l) to D(m)
    for ( auto& ele : _to_l ) {
      int l = ele.first;
      double alpha = ele.second;
      addScaledTo( t, q + l * K, K, alpha );
    }

    // minus P(m) from D(m)
    minusFrom( t, P + m * K, K );
  }
  
  
  inline void update_D( int m, int l, const double *q_l, double alpha )
  {
    // D(m) = D(m) - q(l) * alpha(m,l)
    minusScaledFrom( D + m * K, q + l * K, K, alpha );
    // D(m) = D(m) + q_new(l) * alpha(m,l)
    addScaledTo( D + m * K, q_l, K, alpha );
  }

  inline void altered_D( int m, int l, const double *q_l, double alpha, double *dst )
  {
    copy( dst, D + m * K, K );
    // D(m) = D(m) - q(l) * alpha(m,l)
    minusScaledFrom( dst, q + l * K, K, alpha );
    // D(m) = D(m) + q_new(l) * alpha(m,l)
    addScaledTo( dst, q_l, K, alpha );
  }




  /* ---------- d(n) ---------- */

  // d(n) = sum_l (alpha(i_n,l) - alpha(j_n,l)) * q(l)
  inline void update_d( int n )
  {
    // alias t <- d(n)
    double *t = d + n * K;
    zero( t, K );
    
    { 
      // handle i's side
      int i = patchPairs[n].first;
      auto& _to_l = m_to_l->getToSet( i );
      for ( auto& ele : _to_l ) {
        int l = ele.first;
        double alpha = ele.second;
        addScaledTo( t, q + l * K, K, alpha );
      }
    }

    {
      // handle j's side
      int j = patchPairs[n].first;
      auto& _to_l = m_to_l->getToSet( j );
      for ( auto& ele : _to_l ) {
        int l = ele.first;
        double alpha = ele.second;
        minusScaledFrom( t, q + l * K, K, alpha );
      }
    }
  }

  inline void update_d( int n, int l, const double *q_l, double alpha_i, double alpha_j )
  {
    double *t = d + n * K;
    // d(n) = d(n) - alpha(i,l) * q(l) + alpha(i,l) * q_l
    minusScaledFrom( t, q + l * K, K, alpha_i );
    addScaledTo( t, q_l, K, alpha_i );
    
    // d(n) = d(n) + alpha(j,l) * q(l) - alpha(j,l) * q_l
    addScaledTo( t, q + l * K, K, alpha_j );
    minusScaledFrom( t, q_l, K, alpha_j );
  }

  inline void update_d( int n, int l, const double *q_l, double alpha_i, double alpha_j )
  {
    double *t = d + n * K;
    // d(n) = d(n) - alpha(i,l) * q(l) + alpha(i,l) * q_l
    minusScaledFrom( t, q + l * K, K, alpha_i );
    addScaledTo( t, q_l, K, alpha_i );
    
    // d(n) = d(n) + alpha(j,l) * q(l) - alpha(j,l) * q_l
    addScaledTo( t, q + l * K, K, alpha_j );
    minusScaledFrom( t, q_l, K, alpha_j );
  }

  inline altered_d( int n, int l, const double *q_l, double alpha_i, double alpha_j, double *dst )
  {
    copy( dst, d + n * K );
    
    // d(n) = d(n) - alpha(i,l) * q(l) + alpha(i,l) * q_l
    minusScaledFrom( dst, q + l * K, K, alpha_i );
    addScaledTo( dst, q_l, K, alpha_i );
    
    // d(n) = d(n) + alpha(j,l) * q(l) - alpha(j,l) * q_l
    addScaledTo( dst, q + l * K, K, alpha_j );
    minusScaledFrom( dst, q_l, K, alpha_j );
  }

  

  
  /* ---------- Energy Computation ---------- */

  inline double total_energy()
  {

    double energy_first = 0.0;

    // sum_m D(m)^2
    double *Dp = D;
    for ( int m=0; m<M; m++ ) {
      energy_first += norm2( Dp, K );
      Dp += K;
    }

    double energy_second = 0.0;

    // sum_n d(n)^2 * w(n)
    // where w(n) = w(i_n,j_n)
    double *dp = d;
    for ( int n=0; n<N; n++ ) {
      energy_second += norm2( dp, K ) * w[n];
      dp += K;
    }
    
    return energy_first + energy_second * options.beta;
  }

  /* Restricted Energy on q(l):
   * sum_{m \in l} D(m)^2 + \beta \sum_{n \in l} w(i_n,j_n) * d(n)^2
   */
  inline double restrict_energy( int l )
  {

    auto& _to_m = m_to_l->getFromSet( l );
    auto& _to_n = pair_to_l->getFromSet( l );
    
    
    // energy_first = sum_{m \in l} D(m)^2
    double energy_first = 0.0;
    for ( auto& ele : _to_m ) {
      int m = ele.first;
      energy_first += norm2( D + m * K, K );
    }

    // energy_second = \sum_{n \in l} w(i_n,j_n) * d(n)^2
    double energy_second = 0.0;
    for ( auto& ele : _to_n ) {
      int n = ele.first;
      energy_second =+= norm2( d + n * K, K ) * w[n];
    }

    return energy_first + energy_second * options.beta;
  }

  
  /* Restricted Energy on q(l) replaced with q_l
   * sum_{m \in l} D(m)^2 + \beta \sum_{n \in l} w(i_n,j_n) * d(n)^2
   */
  inline double restrict_energy( int l, const double *q_l )
  {
    auto& _to_m = m_to_l->getFromSet( l );
    auto& _to_n = pair_to_l->getFromSet( l );

    std::unordered_map<int,double> alphas;

    double t[K];

    // energy_first = sum_{m \in l} altered_D(m)^2
    double energy_first = 0.0;
    for ( auto& ele : _to_m ) {
      int m = ele.first;
      double alpha = ele.second;
      // map alpha and m
      alphas[m] = alpha;
      altered_D( m, l, q_l, alpha, t );
      energy_first += norm2( t, K );
    }

    // energy_second = \sum_{n \in l} w(i_n,j_n) * d(n)^2
    double energy_second = 0.0;
    for ( auto& ele : _to_n ) {
      int n = ele.first;
      int i = patchPairs[n].first;
      int j = patchPairs[n].second;
      double alpha_i = 0.0;
      if ( alphas.end() != alphas.find(i) ) {
        alpha_i = alphas[i];
      }
      double alpha_j = 0.0;
      if ( alphas.end() != alphas.find(j) ) {
        alpha_j = alphas[j];
      }
      altered_d( n, l, q_l, alpha_i, alpha_j, t );
      energy_second += norm2( t, K ) * w[n];
    }

    return energy_first + energy_second * options.beta;
  }




  /* ---------- Restricted Derivative ---------- */
  
  inline void restrict_deriv( int l, double *deriv )
  {

    zero( deriv, K );

    std::unordered_map<int,double> alphas;
    
    // deriv = sum_m alpha(l,m) * D(m)
    auto& _to_m = m_to_l->getFromSet( l );
    for ( auto& ele : _to_m ) {
      int m = ele.first;
      double alpha = ele.second;
      // map alpha and m
      alphas[m] = alpha;
      addScaledTo( deriv, D + m * K, K, alpha );
    }
    
    // deriv += beta * sum_n (alpha(i_n,l)-alpha(j_n,l)) * d(n) * w(n)
    for ( auto& ele : _to_n ) {
      int n = ele.first;
      int i = patchPairs[n].first;
      int j = patchPairs[n].second;
      double alpha_i = 0.0;
      if ( alphas.end() != alphas.find(i) ) {
        alpha_i = alphas[i];
      }
      double alpha_j = 0.0;
      if ( alphas.end() != alphas.find(j) ) {
        alpha_j = alphas[j];
      }
      addScaledTo( deriv, d + n * K, K, beta * (alpha_i - alpha_j) * w[n] );
    }
  }


  /* ---------- update q ---------- */

  inline void update_q( int l )
  {
    
    double t0[K];
    double t1[K];
    double t2[K];

    // Get the negative derivative
    restrict_deriv( l, t0 );
    negate( t0, K );

    
    // Line Search Parabola
    
    double t3[K];        
    bool updated = false;
    double a = 1.0; // initial step size
    double dE2 = norm2( t0, K );
    double E0 = restrict_energy( l );

    memcpy( t1, q + l * K, sizeof(double) * K );
    addScaledTo( t1, t0, K, a );
    watershed( t1, t2, K );
    double E_a = restrict_energy( l, t2 );

    
    if ( E_a >= E0 ) {

      // Shrinking Branch
      for ( int i=0; i<40; i++ ) {

        a = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );

        // update E_a
        memcpy( t1, q + l * K, sizeof(double) * K );
        addScaledTo( t1, t0, K, a );
        watershed( t1, t2, K );
        E_a = restrict_energy( l, t2 );


        if ( E_a < E0 ) {
          updated = true;
          break;
        }
      }
    } else { 
      // Expanding Branch
      
      updated = true;
      
      double E_best = E_a;
      for ( int i=0; i<40; i++ ) {
        if ( E_a > E0 - dE2 * a * ( 1 - 0.5 / 2.0 ) ) {
          double b = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );
          
          // update E_a
          memcpy( t1, q + l * K, sizeof(double) * K );
          addScaledTo( t1, t0, K, b );
          watershed( t1, t3, K );
          E_a = restrict_energy( l, t3 );
    
          if ( E_a < E_best ) {
            E_best = E_a;
            memcpy( t2, t3, sizeof(double) * K );
          }
        } else {
          a = a * 2.0;
        }

        memcpy( t1, q + l * K, sizeof(double) * K );
        addScaledTo( t1, t0, K, a );
        watershed( t1, t3, K );
        E_a = restrict_energy( l, t3 );
        
        if ( E_a < E_best ) {
          E_best = E_a;
          memcpy( t2, t3, sizeof(double) * K );
        } else {
          break;
        }

      }
    }

    if ( updated ) {
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;
        update_D( m, l, t2, alpha );
      }
      memcpy( q + l * K, t2, sizeof(double) * K );
    }
    
    
  }


  

public:
  
  void operator()( int M1, int L1,
                   const Bipartite *m_to_l1,
                   const std::vector<std::pair<int,int> > *patchPairs1,
                   const double *w1,
                   const double *P1, double *q1 )
  {

    K = LabelSet::classes;
    L = L1;
    N = static_cast<int>( patchPairs1->size() );
    numL = M1;
    numU = 0;

    P = P1;
    w = w1;
    q = q1;
    d = new double[N*K];
    D = new double[(numL + numU) * K];
    
    m_to_l = m_to_l1;
    patchPairs = patchPairs1;

    
    M = M1;
    L = L1;
    D = new double[M*LabelSet::classes];
    P = P1;
    q = q1;
    forest = forest1;
    m_to_l = m_to_l1;

    // Update D(m)'s
    for ( int m=0; m<numL; m++ ) {
      update_D(m);
    }

    // update d(n)'s
    for ( int n=0; n<N; n++ ) {
      update_d(n);
    }

    
    for ( iter=0; iter<options.maxIter; iter++ ) {

      // Update q(l)'s
      for ( int l=0; l<L; l++ ) {
        update_q(l);
      }

      Info( "Iteration %d - Energy: %.5lf\n", iter, total_energy() );
      
    }

    DeleteToNullWithTestArray( D );
    DeleteToNullWithTestArray( d );
  }

};


int main( int argc, char **argv )
{
  
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (verify.conf)" );
    exit( -1 );
  }

  srand( 345645631 );
  

  env.parse( argv[1] );
  env.Summary();
  
  LabelSet::initialize( env["color-map"] );


  auto img = std::move( cvFeat<HOG>::gen( env["src-img"] ) );
  auto lbl = std::move( cvFeat<BGR>::gen( env["lbl-img"] ) );

  /// 1. Create an random partition of the patches in this image
  std::vector<std::pair<int,int> > training;
  std::vector<std::pair<int,int> > testing;

  for ( int i=7; i<img.rows-7; i++ ) {
    for ( int j=7; j<img.cols-7; j++ ) {
      if ( 0 == ( rand() & 1 ) ) {
        training.push_back( std::make_pair( i, j ) );
      } else {
        testing.push_back( std::make_pair( i, j ) );
      }
    }
  }

  /// 2. train the forest
  std::vector<FeatImage<float>::PatchProxy> l;

  for ( auto& every : training ) {
    l.push_back( img.Spawn( every.first, every.second ) );
  }
  
  timer::tic();
  Forest<SimpleKernel<float> > forest( 10, l, 0.5 );
  Done( "Tree built within %.5lf sec.", timer::utoc() );

  /// 3. Learn the leaf weights
  forest.PrepareWeitghts();
  float feat[img.GetPatchDim()];

  // for ( auto& ele : training ) {
  //   img.FetchPatch( ele.first, ele.second, feat );
  //   forest.learn( feat );
  // }
  // timer::tic();
  // for ( auto& ele : testing ) {
  //   img.FetchPatch( ele.first, ele.second, feat );
  //   forest.learn( feat );
  // }


  for ( int i=0; i<env["learn-img"].length(); i++ ) {
    auto learn_img = std::move( cvFeat<HOG>::gen( env["learn-img"][i] ) );
    for ( int i=7; i<learn_img.rows-7; i++ ) {
      for ( int j=7; j<learn_img.cols-7; j++ ) {
        learn_img.FetchPatch( i, j, feat );
        forest.learn( feat );
      }
    }
  }
  
  forest.FilterWeights( 1 );
  Done( "Weights learned within %.5lf sec.", timer::utoc() );

  // debugging:
  cv::Mat srcimg = cv::imread( env["src-img"] );

  for ( int l=0; l<forest.centers(); l+=100 ) {
    IconList major_leaf( "major leaf node", 13 );
    for ( auto& ele : forest(l).store ) {
      major_leaf.push( srcimg, PatLoc( ele ) );
    }
    major_leaf.display();
    for ( auto& leaf : forest.GetWeights( l ) ) {
      IconList major_leaf( "major leaf node", 13 );
    for ( auto& ele : forest(l).store ) {
      major_leaf.push( srcimg, PatLoc( ele ) );
    }
    major_leaf.display();

    }
    printf( "=============== %ld/%d ===============\n", forest.GetWeights(l).size(), forest.centers() );
    cv::waitKey();
  }

  /// 4. Train Label

  
  std::vector<pair<int,int> > labeled;
  for ( auto& ele : training ) {
    if ( static_cast<double>( rand() ) / RAND_MAX < env["training-label-ratio"].toDouble() ) {
      labeled.push_back( ele );
    }
  }
  
  int M = static_cast<int>( labeled.size() );
  Bipartite m_to_l( M, forest.centers() );
  double *P = new double[ M * LabelSet::classes ];
  double *pP = P;
  int m = 0;
  for ( auto& ele : labeled ) {
    img.FetchPatch( ele.first, ele.second, feat );
    auto res = std::move( forest.query_with_coef( feat ) );
    for ( auto& item : res ) {
      m_to_l.add( m, item.first, item.second );
    }

    const uchar* color = lbl(ele.first,ele.second);
    int classID = LabelSet::GetClass( color[0],
                                      color[1],
                                      color[2] );

    for ( int k=0; k<LabelSet::classes; k++ ) {
      if ( k == classID ) {
        *(pP++) = 1.0;
      } else {
        *(pP++) = 0.0;
      }
    }
    m++;
  }

  Solver solver;
  solver.options.beta = 0.00001;
  solver.options.maxIter = 10;

  double *q = new double[ forest.centers() * LabelSet::classes ];
  double *qp = q;
  for ( int i=0; i<forest.centers() * LabelSet::classes; i++ ) *(qp++) = LabelSet::inv;

  Info( "Solving ..." );
  solver.solve( M, forest.centers(), &forest, &m_to_l, P, q );
  Done( "Solved." );

  qp = q;
  for ( int l=0; l<forest.centers(); l++ ) {
    forest.updateLabelMap( l, qp );
    qp += LabelSet::classes;
  }

  DeleteToNullWithTestArray( q );
  DeleteToNullWithTestArray( P );
  Done( "Label Trained." );

  
  
  
  /// 5. Query
  
  GeoMap geomap = std::move( forest.PatchMatch( img ) );

  // 5.5 Verify PatchMatch is working
  // cv::Mat srcmat = cv::imread( env["src-img"] );
  // ImageViewer srcv( "source", srcmat );
  // ImageViewer refv( "reference", srcmat );

  // srcv.setCallback( [&geomap,&refv,&feat,&img]( int x, int y )
  //                   {
  //                     std::vector<PatLoc> list(1);
                        
  //                     img.FetchPatch( y, x, feat );

  //                     double minDist = 1000.0;
  //                     float feat_c[img.GetPatchDim()];
  //                     for ( auto& ele : geomap(y,x) ) {
  //                       PatLoc loc = ele.apply(y,x);
  //                       img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
  //                       double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
  //                       if ( dist < minDist ) {
  //                         list[0] = loc;
  //                         minDist = dist;
  //                       }
  //                     }

                        
  //                     printf( "----------------------------------------\n" );
  //                     for ( auto& ele : list ) {
  //                       ele.show();
  //                     }
  //                     refv.display( list );
  //                   } );
    
  // while ( 27 != cv::waitKey(30) );


  // 5.6 Check Leaf Node
  // cv::Mat srcmat = cv::imread( env["src-img"] );
  // ImageViewer srcv( "source", srcmat );
  // ImageViewer refv( "reference", srcmat );
  // IconList iconlist( "leaf", 13 );
  // srcv.setCallback( [&geomap,&refv,&forest,&feat,&img,&iconlist,&srcmat]( int x, int y )
  //                   {
  //                     std::vector<PatLoc> list(1);
                        
  //                     img.FetchPatch( y, x, feat );

  //                     double minDist = 1000.0;
  //                     float feat_c[img.GetPatchDim()];
  //                     for ( auto& ele : geomap(y,x) ) {
  //                       PatLoc loc = ele.apply(y,x);
  //                       img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
  //                       double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
  //                       if ( dist < minDist ) {
  //                         list[0] = loc;
  //                         minDist = dist;
  //                       }
  //                     }


  //                     auto res = std::move( forest.query( feat_c ) );
  //                     for ( auto& ele : res ) {
  //                       iconlist.clear();
  //                       for ( auto& item : forest(ele).store ) {
  //                         iconlist.push( srcmat, PatLoc( 0, item.y, item.x, 0.0f, 1.0f ) );
  //                       }
  //                       iconlist.display();
  //                       printVec( &forest(ele).q[0], LabelSet::classes );
  //                       while ( 32 != cv::waitKey(30) );
  //                     }

  //                     refv.display( list );
  //                   } );
    
  // while ( 27 != cv::waitKey(30) );
  



  /// 6. Vote
  
  cv::Mat estimated( geomap.rows, geomap.cols, CV_8UC3 );
  float feat_c[img.GetPatchDim()];  
  float vote[LabelSet::classes];
  for ( int i=0; i<geomap.rows; i++ ) {
    for ( int j=0; j<geomap.cols; j++ ) {
      double minDist = 1000.0;
      PatLoc best;
      img.FetchPatch( i, j, feat );
      
      for ( auto& ele : geomap(i,j) ) {
        PatLoc loc = ele.apply(i,j);
        img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
        double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
        if ( dist < minDist ) {
          minDist = dist;
          best = loc;
        }
      }
      img.FetchPatch( best.y, best.x, best.rotation, best.scale, feat_c );
      auto res = std::move( forest.query( feat_c ) );

      memset( vote, 0, sizeof(float) * LabelSet::classes );
      for ( auto& ele : res ) {
        addto( vote, &forest(ele).q[0], LabelSet::classes );
      }

      float highest = 0.0f;
      int classID = -1;
      for ( int k=0; k<LabelSet::classes; k++ ) {
        if ( vote[k] > highest ) {
          highest = vote[k];
          classID = k;
        }
      }
      
      
      auto& color = LabelSet::GetColor( classID );
      estimated.at<cv::Vec3b>(i,j)[0] = std::get<2>( color );
      estimated.at<cv::Vec3b>(i,j)[1] = std::get<1>( color );
      estimated.at<cv::Vec3b>(i,j)[2] = std::get<0>( color );
      
      
    }
  }

  cv::imshow( "result map", estimated );

  cv::imwrite( "result.png", estimated );
  
  while ( 27 != cv::waitKey(30) );
  
  return 0;
}

