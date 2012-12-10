#include <vector>
#include "LLPack/utils/extio.hpp"
#include "../../data/FeatImage.hpp"
#include "../../interfaces/opencv_aux.hpp"
#include "../forest.hpp"


using namespace PatTk;

int main( int argc, char **argv )
{
  Info( "Loading Forest ..." );
  Forest<SimpleKernel<float> > forest( "forest" );

  
  for ( int i=0; i<forest.centers(); i++ ) {
    for ( auto& ele : forest.GetWeights(i) ) {
      printf( "%d: %d\n", ele.first, ele.second );
    }
    char ch;
    scanf( "%c", &ch );
  }
  
  return 0;
}
