#include <iostream>
#include <algorithm>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int l, a, b, c, d;
	cin >> l >> a >> b >> c >> d;
	int i = l - (a / c);
	if (a % c)
		i--;
	int j = l - (b / d);
	if (b % d)
		j--;
	cout << min(i, j);
}