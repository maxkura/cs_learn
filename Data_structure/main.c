#include <stdio.h>


int main()
{

    int *p;
    p=(int *)malloc(sizeof(int));
    printf("%d",*p);
    return 0;
}