#include <iostream>
#include <queue>
#include <cstring>
#include <vector>
#include <algorithm>
using namespace std;
typedef pair<int, int> pi;

int n, src, dest, m;
bool visited[101];
bool edge[101][101];

int bfs()
{
	memset(visited, false, sizeof(visited));

	queue<pi> q;
	q.push({ src, 0 });
	visited[src] = true;

	while (!q.empty()) {
		pi cur = q.front();
		q.pop();

		if (cur.first == dest)
			return cur.second;

		for (int i = 1; i <= n; i++) {
			if (visited[i])
				continue;
			if (edge[cur.first][i] || edge[i][cur.first]) {
				q.push({ i, cur.second + 1 });
				visited[i] = true;
			}
		}
	}
	return -1;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> src >> dest >> m;

	for (int i = 0; i < m; i++) {
		int v1, v2;
		cin >> v1 >> v2;

		edge[v1][v2] = edge[v2][v1] = true;
	}
	cout << bfs();
}