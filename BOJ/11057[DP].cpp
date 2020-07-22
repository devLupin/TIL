#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int ans= 0, n, DP[1001][10] = { 0, };
	cin >> n;

	for (int i = 0; i < 10; i++)
		DP[1][i] = 1;

	for (int i = 2; i <= n; i++)
		for (int j = 0; j < 10; j++)
			for (int k = j; k < 10; k++)
				DP[i][j] = (DP[i][j] + DP[i - 1][k]) % 10007;

	for (int i = 0; i < 10; i++)
		ans = (ans + DP[n][i]) % 10007;
	cout << ans;
}