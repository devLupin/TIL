#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	int tc; cin >> tc;
	while (tc--) {
		int n;
		long long DP[101] = { 0,1,1,1 };
		cin >> n;
		
		for (int i = 4; i <= n; i++)
			DP[i] = DP[i - 3] + DP[i - 2];
		
		cout << DP[n] << '\n';
	}
}