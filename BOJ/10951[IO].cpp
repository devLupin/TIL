#include<stdio.h>
#pragma warning(disable:4996)
int main(void)
{
	int a, b;
	while (scanf("%d %d", &a, &b) != EOF)
		printf("%d\n", a + b);
}