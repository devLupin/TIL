#include<stdio.h>
int main(void)
{
	int tc; scanf("%d", &tc);
	while (tc--) {
		int a, b; scanf("%d %d", &a, &b);
		printf("%d\n", a + b);
	}
}