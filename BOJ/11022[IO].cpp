#include<stdio.h>
#pragma warning(disable:4996)
int main(void)
{
	int tc, a, b;
	scanf("%d", &tc);
	for (int i = 1; i <= tc; i++) {
		scanf("%d %d", &a, &b);
		printf("Case #%d: %d + %d = %d\n", i, a, b, a + b);
	}
}