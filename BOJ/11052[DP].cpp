#include <iostream>
#include <algorithm>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	int n, arr[1001] = { 0, }, DP[1001] = { 0, };
	cin >> n;
	for (int i = 1; i <= n; i++)
		cin >> arr[i];
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= i; j++)
			DP[i] = max(DP[i], arr[j] + DP[i - j]);
	cout << DP[n];
}