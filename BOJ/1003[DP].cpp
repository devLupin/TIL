#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int DP[41] = { 0, 1, 1 };
	
	for (int i = 3; i <= 40; i++)
		DP[i] = DP[i - 1] + DP[i - 2];

	int tc; cin >> tc;
	while (tc--) {
		int n; cin >> n;

		if (n == 0) cout << 1 << ' ' << 0 << '\n';
		else if (n == 1) cout << 0 << ' ' << 1 << '\n';
		else cout << DP[n - 1] << ' ' << DP[n] << '\n';
	}
}