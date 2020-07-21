#include <iostream>
#include <algorithm>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int tc; cin >> tc;
	while (tc--) {
		int n, cache[2][100001], DP[2][100001] = { 0, };
		cin >> n;

		for (int i = 0; i < 2; i++)
			for (int j = 1; j <= n; j++)
				cin >> cache[i][j];

		DP[0][0] = DP[1][0] = 0;
		DP[0][1] = cache[0][1]; DP[1][1] = cache[1][1];
		for (int i = 2; i <= n; i++) {
			DP[0][i] = max(DP[1][i - 1], DP[1][i - 2]) + cache[0][i];
			DP[1][i] = max(DP[0][i - 1], DP[0][i - 2]) + cache[1][i];
		}
		cout << max(DP[0][n], DP[1][n]) << '\n';
	}
}