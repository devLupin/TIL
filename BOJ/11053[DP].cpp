#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, ans = 0, arr[1001] = { 0, }, DP[1001] = { 0, };
	cin >> n;
	for (int i = 1; i <= n; i++)
		cin >> arr[i];
	
	for (int i = 1; i <= n; i++) {
		int cmp = 0;
		for (int j = 0; j < i; j++) {
			if (arr[i] > arr[j])
				if (cmp < DP[j])
					cmp = DP[j];
		}
		DP[i] = cmp + 1;
		ans = (ans < DP[i]) ? DP[i] : ans;
	}
	cout << ans;
}