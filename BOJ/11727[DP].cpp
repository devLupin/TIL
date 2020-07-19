#include <iostream>
using namespace std;
const int mod = 10007;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, DP[1001] = { 0, 1, 3, };
	cin >> n;
	for (int i = 3; i <= n; i++)
		DP[i] = (DP[i - 1] % mod) + (2 * DP[i - 2] % mod);
	cout << DP[n] % mod;
}