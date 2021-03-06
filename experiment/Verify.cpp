#include <vector>
#include <set>
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



#define DEBUGGING false

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
  const Bipartite *pair_to_l;
  
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
    pair_to_l = nullptr;
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
      int i = (*patchPairs)[n].first;
      auto& _to_l = m_to_l->getToSet( i );
      for ( auto& ele : _to_l ) {
        int l = ele.first;
        double alpha = ele.second;
        addScaledTo( t, q + l * K, K, alpha );
      }
    }

    {
      // handle j's side
      int j = (*patchPairs)[n].second;
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

  inline void altered_d( int n, int l, const double *q_l, double alpha_i, double alpha_j, double *dst )
  {
    copy( dst, d + n * K, K );
    
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
    printf( "numU: %d\n", numU );
    for ( int m=0; m<numU; m++ ) {
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

    // debugging:
    printf( "%.6lf + %.6lf\n", energy_first, energy_second );
    
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
      energy_second += norm2( d + n * K, K ) * w[n];
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
      int i = (*patchPairs)[n].first;
      int j = (*patchPairs)[n].second;
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
    auto& _to_n = pair_to_l->getFromSet( l );
    for ( auto& ele : _to_n ) {
      int n = ele.first;
      int i = (*patchPairs)[n].first;
      int j = (*patchPairs)[n].second;
      double alpha_i = 0.0;
      if ( alphas.end() != alphas.find(i) ) {
        alpha_i = alphas[i];
      }
      double alpha_j = 0.0;
      if ( alphas.end() != alphas.find(j) ) {
        alpha_j = alphas[j];
      }
      addScaledTo( deriv, d + n * K, K, options.beta * (alpha_i - alpha_j) * w[n] );
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
    if ( dE2 < 1e-6 ) {
      return;
    }
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

      auto& _to_m = m_to_l->getFromSet( l );
      std::unordered_map<int,double> alphas;
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;
        alphas[m] = alpha;
        update_D( m, l, t2, alpha );
      }

      auto& _to_n = pair_to_l->getFromSet( l );
      for ( auto& ele : _to_n ) {
        int n = ele.first;
        int i = (*patchPairs)[n].first;
        int j = (*patchPairs)[n].second;
        double alpha_i = 0.0;
        if ( alphas.end() != alphas.find(i) ) {
          alpha_i = alphas[i];
        }
        double alpha_j = 0.0;
        if ( alphas.end() != alphas.find(j) ) {
          alpha_j = alphas[j];
        }
        update_d( n, l, t2, alpha_i, alpha_j );
      }

      memcpy( q + l * K, t2, sizeof(double) * K );

    }
    
  }


  

public:
  
  void operator()( int numL1, int numU1, int L1,
                   const Bipartite *m_to_l1,
                   const Bipartite *pair_to_l1,
                   const std::vector<std::pair<int,int> > *patchPairs1,
                   const double *w1,
                   const double *P1, double *q1 )
  {

    K = LabelSet::classes;
    L = L1;
    N = static_cast<int>( patchPairs1->size() );
    numL = numL1;
    numU = numU1;

    P = P1;
    w = w1;
    q = q1;
    d = new double[N*K];
    D = new double[(numL + numU) * K];

    pair_to_l = pair_to_l1;
    m_to_l = m_to_l1;
    patchPairs = patchPairs1;

    

    // Update D(m)'s
    for ( int m=0; m<numL; m++ ) {
      update_D(m);
    }

    // update d(n)'s
    for ( int n=0; n<N; n++ ) {
      update_d(n);
    }

    
    for (int iter=0; iter<options.maxIter; iter++ ) {

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


void NNGraph( Forest<SimpleKernel<float> > __attribute__((__unused__)) &forest,
              const std::vector<std::pair<int,int> >& training,
              const FeatImage<float>& img,
              std::vector<double>& w,
              std::vector<std::pair<int,int> >& patchPairs )
{
  int M = static_cast<int>( training.size() );
  float feat[img.GetPatchDim()];
  float feat_c[img.GetPatchDim()];

  std::set<std::pair<int,int> > hash;
  
  for ( int i=0; i<M; i++ ) {
    heap<double, int> ranker( env["patch-neighbors"] );
    img.FetchPatch( training[i].first, training[i].second, feat );
    for ( int j=0; j<M; j++ ) {
      if ( static_cast<double>( rand() ) / RAND_MAX > env["nn-sample-ratio"].toDouble() ) {
        continue;
      }
      img.FetchPatch( training[j].first, training[j].second, feat_c );
      if ( i == j ) continue;
      ranker.add( dist_l2( feat, feat_c, img.GetPatchDim() ), j );
    }

    for ( int j=0; j<ranker.len; j++ ) {
      auto p = std::make_pair( i, ranker[j] );
      if ( i > ranker[j] ) {
        p = std::make_pair( ranker[j], i );
      }
      if ( hash.end() == hash.find( p  ) ) {
        if ( ranker(j) < 0.4 ) {
          w.push_back( - 4 * ranker(j) * ranker(j) + 1 );
          patchPairs.push_back( p );
          hash.insert( p );
        }
      }
    }
    progress( i, M, "Patch Nearest Neighbor" );
  }
  printf( "\n" );
}

inline double ExpConv( double dist, double delta )
{
  return exp( - dist / delta );
}

void NNGraph_tree( Forest<SimpleKernel<float> > &forest,
                   const std::vector<std::pair<int,int> >& training,
                   const FeatImage<float>& img,
                   std::vector<double>& w,
                   std::vector<std::pair<int,int> >& patchPairs )
{
  double delta = env["delta"].toDouble();
  
  int M = static_cast<int>( training.size() );
  cv::Mat invIdx( img.rows, img.cols, CV_32SC1 );
  invIdx = cv::Mat::zeros( img.rows, img.cols, CV_32SC1 ) - 1;
  float feat[img.GetPatchDim()];
  float feat_c[img.GetPatchDim()];

  int m = 0;
  for ( auto& ele : training ) {
    invIdx.at<int>( ele.first, ele.second ) = m;
    m++;
  }


  std::set<std::pair<int,int> > hash;
  m = 0;
  for ( auto& ele : training ) {
    img.FetchPatch( ele.first, ele.second, feat );
    auto re = std::move( forest.pull( feat ) );
    heap<double, int> ranker( 10 );
    for ( auto& item : re ) {
      int j = invIdx.at<int>( item.y, item.x );
      if ( -1 != j && j != m ) {
        img.FetchPatch( item.y, item.x, feat_c );
        ranker.add( dist_l2( feat, feat_c, img.GetPatchDim() ),
                    j );
      }
    }


    for ( int j=0; j<ranker.len; j++ ) {
      auto p = std::make_pair( m, ranker[j] );
      if ( m > ranker[j] ) {
        p = std::make_pair( ranker[j], m );
      }
      if ( hash.end() == hash.find( p  ) ) {
        if ( ranker(j) < 0.4 ) {
          w.push_back( ExpConv( ranker(j), delta ) );
          patchPairs.push_back( p );
          hash.insert( p );
        }
      }
    }
    
    m++;
    progress( m, M, "Patch Nearest Neighbor with forest" );

  }
  printf( "\n" );
  printf( "N=%ld\n", w.size() );
}

void NNGraph_patchmatch( Forest<SimpleKernel<float> > &forest,
                         const std::vector<std::pair<int,int> >& training,
                         const FeatImage<float>& img,
                         std::vector<double>& w,
                         std::vector<std::pair<int,int> >& patchPairs )
{
  int M = static_cast<int>( training.size() );
  cv::Mat invIdx( img.rows, img.cols, CV_32SC1 );
  invIdx = cv::Mat::zeros( img.rows, img.cols, CV_32SC1 ) - 1;
  float feat[img.GetPatchDim()];
  float feat_c[img.GetPatchDim()];

  int m = 0;
  for ( auto& ele : training ) {
    invIdx.at<int>( ele.first, ele.second ) = m;
    m++;
  }


  std::set<std::pair<int,int> > hash;
  GeoMap geomap = std::move( forest.PatchMatch( img, 10 ) );
  m = 0;
  for ( auto& ele : training ) {
    img.FetchPatch( ele.first, ele.second, feat );
    for ( auto& item : geomap( ele.first, ele.second ) ) {
      PatLoc loc = item.apply( ele.first, ele.second );
      if ( 0 <= loc.y && loc.y < img.rows && 0 <= loc.x && loc.x < img.cols ) {
        int j = invIdx.at<int>( loc.y, loc.x );
        if ( -1 != j && m != j ) {
          img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
          double dist = dist_l2( feat, feat_c, img.GetPatchDim() );

          auto p = std::make_pair( m, j );
          if ( m > j ) {
            p = std::make_pair( j, m );
          }
          if ( hash.end() == hash.find( p  ) ) {
            if ( dist < 0.4 ) {
              w.push_back( - 4 * dist * dist + 1 );
              patchPairs.push_back( p );
              hash.insert( p );
            }
          }
        }
      }
    }
    m++;
    progress( m, M, "Patch Nearest Neighbor with forest" );
  }
  printf( "\n" );
  printf( "N=%ld\n", w.size() );
}


// for debugging
void CheckGraph( const std::vector<std::pair<int,int> >& patchPairs,
                 const std::vector<std::pair<int,int> >& training,
                 const FeatImage<uchar> &lbl,
                 const FeatImage<float> &img ) {
  for ( auto& ele : patchPairs ) {
    int i = ele.first;
    int j = ele.second;
    int y0 = training[i].first;
    int x0 = training[i].second;
    int y1 = training[j].first;
    int x1 = training[j].second;
    const uchar *c0 = lbl(y0,x0);
    const uchar *c1 = lbl(y1,x1);
    int label0 = LabelSet::GetClass( c0[0], c0[1], c0[2] );
    int label1 = LabelSet::GetClass( c1[0], c1[1], c1[2] );
    float feat[img.GetPatchDim()];
    if ( label0 != label1 ) {
      Info( "Bad pair: (%d,%d) vs (%d,%d)", y0, x0, y1, x1 );
      printf( "feature: " );
      img.FetchPatch( y0, x0, feat );
      printVec( feat, img.GetPatchDim() );
      printf( "feature: " );
      img.FetchPatch( y1, x1, feat );
      printVec( feat, img.GetPatchDim() );
      char ch;
      scanf( "%c", &ch );
    }
  }
}




void RandomPartition( const FeatImage<uchar> &lbl,
                      const std::vector<std::pair<int,int> > &org,
                      std::vector<std::pair<int,int> > &a,
                      std::vector<std::pair<int,int> > &b,
                      float ratio=0.5 )
{

  std::vector<std::pair<int,int> > classified[LabelSet::classes];

  for ( auto& ele : org ) {
    const uchar* color = lbl(ele.first,ele.second);
    int classID = LabelSet::GetClass( color[0],
                                      color[1],
                                      color[2] );
    classified[classID].push_back( ele );
  }
  
  for ( auto& item : classified ) {
    int n = static_cast<int>( item.size() );
    int k = static_cast<int>( n * ratio );
    std::vector<int> selected = rndgen::randperm( n, k );
    std::vector<int> mask( item.size(), 0 );
    for ( auto& ele : selected ) {
      mask[ele] = 1;
      a.push_back( item[ele] );
    }

    for ( int i=0; i<n; i++ ) {
      if ( 0 == mask[i] ) {
        b.push_back( item[i] );
      }
    }
  }
}



void PartitionTrainingTesting( const FeatImage<uchar> &lbl,
                               std::vector<std::pair<int,int> > &training,
                               std::vector<std::pair<int,int> > &testing,
                               float ratio=0.5 )
{

  
  std::vector<std::pair<int,int> > org;
  
  for ( int i=0; i<lbl.rows; i++ ) {
    for ( int j=0; j<lbl.cols; j++ ) {
      org.push_back( std::make_pair( i, j ) );
    }
  }
  RandomPartition( lbl, org, training, testing, ratio );
}


inline void PartitionLabeledUnlabeled( const FeatImage<uchar> &lbl,
                                const std::vector<std::pair<int,int> > &training,
                                std::vector<std::pair<int,int> > &labeled,
                                std::vector<std::pair<int,int> > &unlabeled,
                                float ratio=0.5 )
{
  RandomPartition( lbl, training, labeled, unlabeled, ratio );
}

inline cv::Mat genSpotGraph( const std::vector<std::pair<int,int> >& lst, int height, int width )
{
  cv::Mat canvas = cv::Mat::zeros( height, width, CV_8UC1 );
  for ( auto& ele : lst ) {
    canvas.at<uchar>( ele.first, ele.second ) = 255;
  }
  return canvas;
}


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
  

  auto img = std::move( cvFeat<BGR_FLOAT>::gen( env["src-img"] ) );
  img.SetPatchSize(1);
  img.SetPatchStride(1);
  auto lbl = std::move( cvFeat<BGR>::gen( env["lbl-img"] ) );
  lbl.SetPatchSize(1);
  lbl.SetPatchStride(1);

  /// 1. Create an random partition of the patches in this image
  std::vector<std::pair<int,int> > training;
  std::vector<std::pair<int,int> > testing;


  
  PartitionTrainingTesting( lbl, training, testing, env["training-ratio"].toDouble() );
  cv::imwrite( strf( "%s/sampled_train.png", env["output-dir"].c_str() ), genSpotGraph( training,
                                                                                        img.rows,
                                                                                        img.cols ) );
  

  
  // Partition the training set into labeld and unlabeled
  // and adjust the order
  std::vector<pair<int,int> > labeled;
  std::vector<pair<int,int> > unlabeled;

  PartitionLabeledUnlabeled( lbl, training, labeled, unlabeled, env["training-label-ratio"].toDouble() );
  cv::imwrite( strf( "%s/sampled_labeled.png", env["output-dir"].c_str() ), genSpotGraph( labeled,
                                                                                          img.rows,
                                                                                          img.cols ) );
  cv::imwrite( strf( "%s/sampled_unlabeled.png", env["output-dir"].c_str() ), genSpotGraph( unlabeled,
                                                                                            img.rows,
                                                                                            img.cols ) );

  printf( "labeled: %ld\n", labeled.size() );
    
  // for ( auto& ele : training ) {
  //   if ( static_cast<double>( rand() ) / RAND_MAX < env["training-label-ratio"].toDouble() ) {
  //     labeled.push_back( ele );
  //   } else {
  //     unlabeled.push_back( ele );
  //   }
  // }
  
  training = labeled;
  training.insert( training.end(), unlabeled.begin(), unlabeled.end() );
  
  int numL = static_cast<int>( labeled.size() );
  int numU = static_cast<int>( unlabeled.size() );
  int M = numU + numL;

  /// 2. train the forest
  std::vector<FeatImage<float>::PatchProxy> l;
  
  for ( auto& every : training ) {
    l.push_back( img.Spawn( every.first, every.second ) );
  }
  
  timer::tic();
  Forest<SimpleKernel<float> > forest( env["num-of-trees"], l, env["each-tree-accounts-for"].toDouble() );
  Done( "Tree built within %.5lf sec.", timer::utoc() );

  printf( "centers: %d\n", forest.centers() );
  

  /// 3. Construct Nearest Neightbor Pairs
  
  std::vector<double> w;
  std::vector<std::pair<int,int> > patchPairs;
  float feat[img.GetPatchDim()];
  float feat_c[img.GetPatchDim()];
  if ( ! DEBUGGING ) {

    NNGraph( forest, training, img, w, patchPairs );
    // NNGraph_tree( forest, training, img, w, patchPairs );
    // NNGraph_patchmatch( forest, training, img, w, patchPairs );
    CheckGraph( patchPairs, training, lbl, img );
    
    WITH_OPEN( out, "interpatch.dat", "w" );
    int len = static_cast<int>( w.size() );
    fwrite( &len, sizeof(int), 1, out );
    for ( int i=0; i<len; i++ ) {
      fwrite( &patchPairs[i].first, sizeof(int), 1, out );
      fwrite( &patchPairs[i].second, sizeof(int), 1, out );
      fwrite( &w[i], sizeof(double), 1, out );
    }
    END_WITH( out );
  } else {
    WITH_OPEN( in, "interpatch.dat", "r" );
    int len = 0;
    fread( &len, sizeof(int), 1, in );
    patchPairs.resize( len );
    w.resize( len );
    for ( int i=0; i<len; i++ ) {
      fread( &patchPairs[i].first, sizeof(int), 1, in );
      fread( &patchPairs[i].second, sizeof(int), 1, in );
      fread( &w[i], sizeof(double), 1, in );
    }
    END_WITH( in );
  }
  
  
  
  /// 4. Train Label

  Bipartite m_to_l( M, forest.centers() );
  double *P = new double[ numL * LabelSet::classes ];
  double *pP = P;
  int m = 0;

  std::set<int> touched;
  
  for ( auto& ele : labeled ) {
    img.FetchPatch( ele.first, ele.second, feat );
    auto res = std::move( forest.query_with_coef( feat ) );
    for ( auto& item : res ) {
      m_to_l.add( m, item.first, item.second );
      touched.insert( item.first );
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

  // untouched pixels
  std::vector<std::pair<int,int> > untouched;
  for ( int i=0; i<img.rows; i++ ) {
    for ( int j=0; j<img.cols; j++ ) {
      img.FetchPatch( i, j, feat );
      auto res = std::move( forest.query_with_coef( feat ) );
      int touch = 0;
      for ( auto& item : res ) {
        if ( touched.end() != touched.find( item.first ) ) {
          touch++;
        }
      }
      if ( touch < 4 ) {
        untouched.push_back( std::make_pair( i, j ) );
      }
    }
  }
  cv::imwrite( strf( "%s/untouched.png", env["output-dir"].c_str() ), genSpotGraph( untouched,
                                                                                    img.rows,
                                                                                    img.cols ) );


  Bipartite pair_to_l( static_cast<int>( patchPairs.size() ), forest.centers() );
  int n = 0;
  for ( auto& ele : patchPairs ) {
    std::set<int> hash;
    {
      int i = ele.first;
      auto& _to_l = m_to_l.getToSet( i );
      for ( auto& item : _to_l ) {
        if ( hash.end() == hash.find( item.first ) ) {
          pair_to_l.add( n, item.first, 1.0 );
          hash.insert( item.first );
        }
      }
    }

    {
      int j = ele.second;
      auto& _to_l = m_to_l.getToSet( j );
      for ( auto& item : _to_l ) {
        if ( hash.end() == hash.find( item.first ) ) {
          pair_to_l.add( n, item.first, 1.0 );
          hash.insert( item.first );
        }
      }
    }
    n++;
  }
   
  
  Solver solve;
  solve.options.beta = env["beta"].toDouble();
  solve.options.maxIter = env["max-iter"];
  

  // initialization of q
  double *q = new double[ forest.centers() * LabelSet::classes ];
  double *qp = q;
  // uniform initialization of q
  for ( int i=0; i<forest.centers() * LabelSet::classes; i++ ) *(qp++) = LabelSet::inv;
  // special initialization of q
  {
    double t[LabelSet::classes];
    qp = q;
    for ( int l=0; l<forest.centers(); l++ ) {
      auto& _to_m = m_to_l.getFromSet( l );
      int count = 0;
      zero( t, LabelSet::classes );
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        if ( m < numL ) {
          addto( t, P + m * LabelSet::classes, LabelSet::classes );
          count++;
        }
      }
      if ( count > 0 ) {
        scale( t, LabelSet::classes, 1.0 / count );
        copy( qp, t, LabelSet::classes );
      }
      qp += LabelSet::classes;
    }
  }

  Info( "Solving ..." );

  solve( numL, numU, forest.centers(),
         &m_to_l,
         &pair_to_l,
         &patchPairs,
         &w[0], P, q );
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

  int correct = 0;
  cv::Mat estimated( geomap.rows, geomap.cols, CV_8UC3 );
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

      const uchar *trueColor = lbl(i,j);
      if ( classID == LabelSet::GetClass( trueColor[0],
                                          trueColor[1],
                                          trueColor[2] ) ) {
        correct++;
      }
      
      auto& color = LabelSet::GetColor( classID );
      estimated.at<cv::Vec3b>(i,j)[0] = std::get<2>( color );
      estimated.at<cv::Vec3b>(i,j)[1] = std::get<1>( color );
      estimated.at<cv::Vec3b>(i,j)[2] = std::get<0>( color );
      
      
    }
  }

  cv::imshow( "result map", estimated );

  cv::imwrite( strf( "%s/result.png", env["output-dir"].c_str() ), estimated );

  printf( "correctness: %d/%d\n", correct, img.rows * img.cols );
  
  while ( 27 != cv::waitKey(30) );
  
  return 0;
}

