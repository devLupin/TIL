#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

int n, m;
bool edge[101][101];
bool visited[101];

int bfs()
{
	memset(visited, 0, sizeof(visited));

	int ret = 0;
	queue<int> q;
	q.push(1);
	visited[1] = true;

	while (!q.empty()) {
		int cur = q.front();
		q.pop();

		for (int i = 1; i <= n; i++) {
			if (visited[i])
				continue;

			if (edge[cur][i] || edge[i][cur]) {
				q.push(i);
				visited[i] = true;
				ret++;
			}
		}
	}
	return ret;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> m;
	for (int i = 0; i < m; i++) {
		int v1, v2;
		cin >> v1 >> v2;

		edge[v1][v2] = edge[v2][v1] = true;
	}

	cout << bfs();
}