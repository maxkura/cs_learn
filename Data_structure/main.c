#include  <stdio.h>
#include  <math.h>
typedef struct fushu
{
    float x, y;
}fushu;
void create(float a, float b, fushu& z)
{
    z3.x = a; z.y = b;
};
void add(fushu z1, fushu z2, fushu& z3)
{
    z3.x = z1.x + z2.x;
    z3.y = z1.y + z2.y;
};
void print_fushu(fushu f)
{
    printf("real part and ima part is :%f %f\n", f.x, f.y);
}

long long factorial_recursive(int n) {
    if (n < 0) {
        return 0;  // 负数阶乘未定义
    }
    if (n == 0 || n == 1) {
        return 1;  // 0! = 1, 1! = 1
    }
    return (n * factorial_recursive(n - 1));
}

int main()
{
    fushu ss1;
    fushu ss2;
    fushu ss3;
    float aa, bb;
    printf("input two intger:\n");
    scanf_s("%f %f", &aa, &bb);
     create(aa, bb, ss1);
    printf("input two intger:\n");
  //  printf("再输入两个整数");
    scanf_s("%f %f", &aa, &bb);
    create(aa, bb, ss2);
    add(ss1, ss2, ss3);


    print_fushu(ss1);
    print_fushu(ss2);
    print_fushu(ss3);
    //以下演示n!
    int n; 
    printf("input one intger:\n");
    scanf_s("%d", &n);
    printf("%d！is :%lld\n",n, factorial_recursive(n));

    return 1;
}