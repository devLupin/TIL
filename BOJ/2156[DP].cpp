#include <iostream>
#define MAX(A,B) ((A) > (B)) ? (A) : (B)
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, ans = -1, arr[10003] = { 0, }, DP[10003] = { 0, };
	cin >> n;

	for (int i = 3; i < n+3; i++)
		cin >> arr[i];

	for (int i = 3; i < n + 3; i++) {
		DP[i] = MAX(DP[i - 3] + arr[i - 1] + arr[i], DP[i - 2] + arr[i]);
		DP[i] = MAX(DP[i - 1], DP[i]);	// µÎ Ä­ ÀÌ»ó ¶Ù¾úÀ» °¡´É¼º
		ans = MAX(ans, DP[i]);
	}

	cout << ans;
}