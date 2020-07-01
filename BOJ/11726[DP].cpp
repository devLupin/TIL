#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int DP[1001] = { 0, 1, 2 };
	for (int i = 3; i <= 1000; i++)
		DP[i] = (DP[i - 1] + DP[i - 2]) % 10007;

	int n; cin >> n;
	cout << DP[n];
}