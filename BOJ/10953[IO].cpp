#include<stdio.h>
int main(void)
{
	int tc;
	char a, b, c;
	scanf("%d", &tc);
	scanf("%c", &c);
	while (tc--) {
		scanf("%c %c %c", &a, &c, &b);
		scanf("%c", &c);
		printf("%d\n", (a-'0') + (b-'0'));
	}
}