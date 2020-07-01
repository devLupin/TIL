#include <iostream>
#define MIN(A,B) ((A) < (B) ? (A) : (B))
using namespace std;

int cost[1001][3], DP[1001][3], n;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n;
	for (int i = 1; i <= n; i++)
		cin >> cost[i][0] >> cost[i][1] >> cost[i][2];

	for (int i = 0; i < 3; i++)
		DP[1][i] = cost[1][i];

	for (int node = 2; node <= n; node++) {
		DP[node][0] = MIN(DP[node - 1][1], DP[node - 1][2]) + cost[node][0];
		DP[node][1] = MIN(DP[node - 1][0], DP[node - 1][2]) + cost[node][1];
		DP[node][2] = MIN(DP[node - 1][0], DP[node - 1][1]) + cost[node][2];
	}

	int ans = DP[n][0];
	for (int i = 1; i < 3; i++)
		ans = (ans < DP[n][i]) ? ans : DP[n][i];

	cout << ans;
}