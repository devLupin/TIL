#include<iostream>
using namespace std;

int max(int a, int b) {
	return (a > b) ? a : b;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, ans = -1, arr[100000] = { 0, }, DP[100000];
	cin >> n;

	for (int i = 0; i < n; i++)
		cin >> arr[i];
	DP[0] = ans = arr[0];
	for (int i = 1; i < n; i++) {
		DP[i] = max(DP[i - 1] + arr[i], arr[i]);
		ans = max(DP[i], ans);
	}
	cout << ans;
}