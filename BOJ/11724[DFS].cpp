#include <iostream>
#include <cstring>
using namespace std;

int n, m;
bool edge[1001][1001];
bool node[1001];

void dfs(int cur)
{
	node[cur] = true;
	for (int i = 1; i <= n; i++)
		if (!node[i] && edge[cur][i])
			dfs(i);
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	memset(edge, false, sizeof(edge));

	cin >> n >> m;
	for (int i = 0; i < m; i++) {
		int u, v; cin >> u >> v;
		edge[u][v] = edge[v][u] = true;
	}

	int ans = 0;
	memset(node, false, sizeof(node));
	for(int i=1; i<=n; i++)
		if (!node[i]) {
			ans++;
			dfs(i);
		}

	cout << ans;
}