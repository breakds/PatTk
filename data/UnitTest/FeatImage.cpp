#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../FeatImage.hpp"
#include "../../interfaces/opencv_aux.hpp"
#include "../../query/tree.hpp"



using namespace PatTk;
int main()
{

  srand(0);
  // std::vector<std::string> imgList = std::move( readlines( "camvid_mid/small.txt" ) );
  // auto album = cvAlbum<HOG>::gen( path::FFFL( "camvid_mid", imgList, ".png" ) );

  auto img = cvFeat<HOG>::gen( "camvid_mid/Seq05VD_f00810.png", 3, 0.7 );

  // std::vector<FeatImage<float>::PatchProxy> l;

  // for ( int i=0; i<img.rows; i++ ) {
  //   for ( int j=0; j<img.cols; j++ ) {
  //     l.push_back( img.Spawn( i, j ) );
  //   }
  // }

  // int idx[l.size()];

  // for ( int i=0, end=static_cast<int>( l.size() ); i < end; i++ ) {
  //   idx[i] = i;
  // }

  // Tree<SimpleKernel<float> > tree( l, idx, l.size() );

  // tree.write( "tree.dat" );





  auto tree = Tree<SimpleKernel<float> >::read( "tree.dat" );
  
  FeatImage<float>::PatchProxy p = img.Spawn( 180, 20 );

  const std::vector<LocInfo> &loc = tree->query( p );

  for ( auto& ele : loc ) {
    printf( "%d, %d\n", ele.y, ele.x );
  }


  

  return 0;
}





