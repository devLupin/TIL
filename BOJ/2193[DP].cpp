#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n;
	long long DP[91];

	cin >> n;
	DP[1] = DP[2] = 1;

	for (int i = 3; i <= n; i++)
		DP[i] = DP[i - 1] + DP[i - 2];

	cout << DP[n];
}