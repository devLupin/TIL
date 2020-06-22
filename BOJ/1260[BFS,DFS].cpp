#include <iostream>
#include <queue>
#include <cstring>

using namespace std;

int n, m, v;
bool edge[1001][1001];
bool visited[1001];

void bfs() 
{
	queue<int> q;
	q.push(v);
	visited[v] = true;

	while (!q.empty()) {
		int cur = q.front();
		q.pop();

		cout << cur << " ";

		for (int dir = 1; dir <= n; dir++) {
			if (visited[dir])
				continue;
			if (edge[cur][dir] || edge[dir][cur]) {
				q.push(dir);
				visited[dir] = true;
			}
		}
	}
}

void dfs(int cur)
{
	cout << cur << " ";
	visited[cur] = true;

	for (int dir = 1; dir <= n; dir++) {
		if (visited[dir])
			continue;
		if (edge[cur][dir] || edge[dir][cur]) {
			visited[dir] = true;
			dfs(dir);
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> m >> v;

	for (int i = 0; i < m; i++) {
		int v1, v2;
		cin >> v1 >> v2;

		edge[v1][v2] = edge[v2][v1] = true;
	}

	memset(visited, 0, sizeof(visited));
	dfs(v);
	cout << "\n";
	memset(visited, 0, sizeof(visited));
	bfs();

	return 0;
}