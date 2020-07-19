#include <iostream>
using namespace std;
typedef pair<int, int> pi;

int max(int a, int b) { return a > b ? a : b; }

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, DP[17] = { 0, };
	pi record[17] = { {0,0}, };
	cin >> n;
	for (int i = 1; i <= n; i++) {
		int t, p; cin >> t >> p;
		record[i] = { t,p };
		DP[i] = p;
	}

	for (int i = n; i >= 1; i--) {
		pi cur = record[i];
		int next = i + cur.first;
		DP[i] = (next > n + 1) ? DP[i + 1] : max(DP[i + 1], DP[next] + cur.second);
	}
	cout << DP[1];
}