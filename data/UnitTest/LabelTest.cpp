#include "../Label.hpp"


using namespace PatTk;

int main()
{
  LabelSet::initialize( "colormap.txt" );

  LabelSet::Summary();

  printf( "%d\n", LabelSet::GetClass( 192,192,127 ) );
  return 0;
}
